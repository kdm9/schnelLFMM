use ndarray::Array2;
use std::sync::Mutex;

use crate::bed::{decode_bed_chunk_into, decode_raw_bed_chunk_into, BedFile, SnpNorm, SubsetSpec};

/// A pre-allocated buffer for a chunk of decoded SNP data.
pub struct SnpBlock {
    /// Decoded genotype data: n_samples × chunk_capacity (pre-allocated).
    /// Only the first `n_cols` columns contain valid data.
    pub data: Array2<f64>,
    /// Actual number of columns used (last chunk may be smaller than capacity).
    pub n_cols: usize,
    /// Sequential chunk index (0, 1, 2, ...) assigned by the IO thread.
    pub seq: usize,
    /// Raw packed bytes copied from mmap by the IO thread.
    /// Workers decode from this into `data` before calling process_fn.
    pub raw: Vec<u8>,
}

/// Imputation strategy for the streaming decode step.
#[derive(Clone)]
pub enum ImputeConfig {
    /// Current behaviour: fill NaN with per-SNP mean (zero after centering).
    Mean,
    /// NMF with pre-computed H: fill NaN with W @ H_chunk before centering.
    /// W is n × K, h_full is K × p (or K × p_est), indexed by chunk offset.
    NmfInRam {
        w: Array2<f64>,
        h_full: Array2<f64>,
    },
    /// NMF on-the-fly: compute H_chunk = max(0, W_pinv @ Y_filled) within each chunk,
    /// then fill NaN with W @ H_chunk. Used in the testing pass.
    NmfOnTheFly {
        w: Array2<f64>,
        w_pinv: Array2<f64>,
    },
}

/// Per-worker accumulator for forward passes that sum partial results.
///
/// Each worker thread locks its own buffer (indexed by worker_id) before writing.
/// After all workers finish, call `.sum()` to merge all buffers.
pub struct PerWorkerAccumulator {
    buffers: Vec<Mutex<Array2<f64>>>,
}

impl PerWorkerAccumulator {
    /// Create `n_workers` zero-initialized accumulators of the given shape.
    pub fn new(n_workers: usize, shape: (usize, usize)) -> Self {
        let buffers = (0..n_workers)
            .map(|_| Mutex::new(Array2::<f64>::zeros(shape)))
            .collect();
        PerWorkerAccumulator { buffers }
    }

    /// Get a mutable reference to the buffer for `worker_id`.
    ///
    /// Returns a `MutexGuard` that dereferences to `&mut Array2<f64>`.
    pub fn get_mut(&self, worker_id: usize) -> std::sync::MutexGuard<'_, Array2<f64>> {
        self.buffers[worker_id].lock().unwrap()
    }

    /// Consume the accumulator and return the element-wise sum of all buffers.
    pub fn sum(self) -> Array2<f64> {
        let mut buffers: Vec<Array2<f64>> = self
            .buffers
            .into_iter()
            .map(|m| m.into_inner().unwrap())
            .collect();
        let mut total = buffers.pop().expect("PerWorkerAccumulator must have >= 1 buffer");
        for buf in buffers {
            total += &buf;
        }
        total
    }
}

/// Expand a `SubsetSpec` into a `Vec<usize>` of SNP indices.
pub fn subset_indices(subset: &SubsetSpec, n_snps: usize) -> Vec<usize> {
    match subset {
        SubsetSpec::All => (0..n_snps).collect(),
        SubsetSpec::Rate(rate) => {
            let step = (1.0 / rate).ceil() as usize;
            (0..n_snps).step_by(step).collect()
        }
        SubsetSpec::Indices(indices) => indices.clone(),
    }
}

/// Copy raw packed bytes for `chunk_indices` from mmap into `dst`.
///
/// SNP `i` within the chunk is stored at `dst[i * bps .. (i+1) * bps]`.
/// For contiguous index ranges (the common case with SubsetSpec::All) this
/// collapses into a single memcpy.
fn copy_raw_chunk(mmap_data: &[u8], bps: usize, chunk_indices: &[usize], dst: &mut [u8]) {
    let n_cols = chunk_indices.len();
    let contiguous = n_cols > 1
        && chunk_indices[n_cols - 1] - chunk_indices[0] == n_cols - 1;

    if contiguous || n_cols == 1 {
        let start = chunk_indices[0] * bps;
        let len = n_cols * bps;
        dst[..len].copy_from_slice(&mmap_data[start..start + len]);
    } else {
        for (i, &snp_idx) in chunk_indices.iter().enumerate() {
            let src_start = snp_idx * bps;
            let dst_start = i * bps;
            dst[dst_start..dst_start + bps]
                .copy_from_slice(&mmap_data[src_start..src_start + bps]);
        }
    }
}

/// Stream SNP chunks through a pool of pre-allocated SnpBlock buffers,
/// with 1 IO thread and `n_workers` worker threads connected via crossbeam channels.
///
/// Architecture:
/// - **IO thread**: copies raw packed bytes from the mmap sequentially into buffers.
/// - **Worker threads**: decode the raw 2-bit→f64 data, then call `process_fn`.
///
/// Uses `std::thread::scope` so that `process_fn` can borrow from the caller's stack.
pub fn parallel_stream<F>(
    bed: &BedFile,
    subset: &SubsetSpec,
    chunk_size: usize,
    n_workers: usize,
    norm: SnpNorm,
    impute: ImputeConfig,
    process_fn: F,
) where
    F: Fn(usize, &SnpBlock) + Send + Sync,
{
    let n_workers = n_workers.max(1);
    let indices = subset_indices(subset, bed.n_snps);
    let n_output_samples = bed.n_samples;
    let n_physical_samples = bed.n_physical_samples;
    let sample_keep = bed.sample_keep.as_deref();
    let bps = bed.bytes_per_snp();

    let pool_size = n_workers + 1;
    let raw_buf_size = chunk_size * bps;

    let local_indices: Vec<usize> = (0..chunk_size).collect();

    let (free_tx, free_rx) = crossbeam_channel::bounded::<SnpBlock>(pool_size);
    let (filled_tx, filled_rx) = crossbeam_channel::bounded::<SnpBlock>(pool_size);

    for _ in 0..pool_size {
        free_tx
            .send(SnpBlock {
                data: Array2::<f64>::zeros((n_output_samples, chunk_size)),
                n_cols: 0,
                seq: 0,
                raw: vec![0u8; raw_buf_size],
            })
            .unwrap();
    }

    let mmap_data = &bed.mmap[3..];
    let impute_ref = &impute;

    std::thread::scope(|s| {
        s.spawn(|| {
            for (seq, chunk_indices) in indices.chunks(chunk_size).enumerate() {
                let mut block = free_rx.recv().unwrap();
                let n_cols = chunk_indices.len();
                copy_raw_chunk(mmap_data, bps, chunk_indices, &mut block.raw);
                block.n_cols = n_cols;
                block.seq = seq;
                filled_tx.send(block).unwrap();
            }
            drop(filled_tx);
        });

        for worker_id in 0..n_workers {
            let filled_rx = filled_rx.clone();
            let free_tx = free_tx.clone();
            let process_fn = &process_fn;
            let local_indices = &local_indices;
            s.spawn(move || {
                while let Ok(mut block) = filled_rx.recv() {
                    let n_cols = block.n_cols;
                    let n_out = n_output_samples;

                    match impute_ref {
                        ImputeConfig::Mean => {
                            let out_view = block.data.slice_mut(ndarray::s![.., ..n_cols]);
                            decode_bed_chunk_into(
                                &block.raw,
                                bps,
                                n_physical_samples,
                                &local_indices[..n_cols],
                                out_view,
                                norm,
                                sample_keep,
                                None,
                            );
                        }
                        ImputeConfig::NmfInRam { w, h_full } => {
                            let start = block.seq * chunk_size;
                            let h_chunk = h_full.slice(ndarray::s![.., start..start + n_cols]);
                            let fill_values = w.dot(&h_chunk);
                            let out_view = block.data.slice_mut(ndarray::s![.., ..n_cols]);
                            decode_bed_chunk_into(
                                &block.raw,
                                bps,
                                n_physical_samples,
                                &local_indices[..n_cols],
                                out_view,
                                norm,
                                sample_keep,
                                Some(fill_values.view()),
                            );
                        }
                        ImputeConfig::NmfOnTheFly { w, w_pinv } => {
                            // Pass 1: raw decode → fill NaN with mean → compute H_chunk
                            {
                                let out = block.data.slice_mut(ndarray::s![.., ..n_cols]);
                                decode_raw_bed_chunk_into(
                                    &block.raw,
                                    bps,
                                    n_physical_samples,
                                    &local_indices[..n_cols],
                                    out,
                                    sample_keep,
                                );
                            }
                            // Fill NaN with column means
                            for col in 0..n_cols {
                                let mut sum = 0.0;
                                let mut n_obs = 0u32;
                                for row in 0..n_out {
                                    let v = block.data[(row, col)];
                                    if !v.is_nan() {
                                        sum += v;
                                        n_obs += 1;
                                    }
                                }
                                let mean = if n_obs > 0 { sum / n_obs as f64 } else { 0.0 };
                                for row in 0..n_out {
                                    if block.data[(row, col)].is_nan() {
                                        block.data[(row, col)] = mean;
                                    }
                                }
                            }
                            // H_chunk = max(0, W_pinv @ Y_filled)
                            let y_view = block.data.slice(ndarray::s![.., ..n_cols]);
                            let h_chunk_raw = w_pinv.dot(&y_view);
                            let h_chunk: Array2<f64> = h_chunk_raw.mapv(|v| v.max(0.0));
                            let fill_values = w.dot(&h_chunk);

                            // Pass 2: final decode with NMF fill (overwrites block.data)
                            let out2 = block.data.slice_mut(ndarray::s![.., ..n_cols]);
                            decode_bed_chunk_into(
                                &block.raw,
                                bps,
                                n_physical_samples,
                                &local_indices[..n_cols],
                                out2,
                                norm,
                                sample_keep,
                                Some(fill_values.view()),
                            );
                        }
                    }

                    process_fn(worker_id, &block);
                    let _ = free_tx.send(block);
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::sync::Mutex;

    #[test]
    fn test_mutex_row_write_basic() {
        let mut arr = Array2::<f64>::zeros((4, 3));
        let mtx = Mutex::new(&mut arr);

        // Write to rows 1..3
        let src = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        {
            let mut guard = mtx.lock().unwrap();
            guard.slice_mut(ndarray::s![1..3, ..]).assign(&src);
        }

        assert_eq!(arr[(0, 0)], 0.0);
        assert_eq!(arr[(1, 0)], 1.0);
        assert_eq!(arr[(1, 2)], 3.0);
        assert_eq!(arr[(2, 0)], 4.0);
        assert_eq!(arr[(2, 2)], 6.0);
        assert_eq!(arr[(3, 0)], 0.0);
    }

    #[test]
    fn test_mutex_row_write_multithreaded() {
        let n_threads = 4;
        let rows_per_thread = 10;
        let cols = 5;
        let total_rows = n_threads * rows_per_thread;
        let mut arr = Array2::<f64>::zeros((total_rows, cols));

        {
            let mtx = Mutex::new(&mut arr);
            std::thread::scope(|s| {
                for t in 0..n_threads {
                    let mtx = &mtx;
                    s.spawn(move || {
                        let start = t * rows_per_thread;
                        let block = Array2::from_shape_fn(
                            (rows_per_thread, cols),
                            |(r, c)| ((start + r) * cols + c) as f64,
                        );
                        let mut guard = mtx.lock().unwrap();
                        guard
                            .slice_mut(ndarray::s![start..start + rows_per_thread, ..])
                            .assign(&block);
                    });
                }
            });
        }

        for r in 0..total_rows {
            for c in 0..cols {
                assert_eq!(arr[(r, c)], (r * cols + c) as f64);
            }
        }
    }

    #[test]
    fn test_mutex_slice_write_multithreaded() {
        let n_threads = 4;
        let elems_per_thread = 10;
        let total = n_threads * elems_per_thread;
        let mut data = vec![0.0f64; total];

        {
            let mtx = Mutex::new(&mut data[..]);
            std::thread::scope(|s| {
                for t in 0..n_threads {
                    let mtx = &mtx;
                    s.spawn(move || {
                        let start = t * elems_per_thread;
                        let values: Vec<f64> =
                            (start..start + elems_per_thread).map(|i| i as f64).collect();
                        let mut guard = mtx.lock().unwrap();
                        guard[start..start + elems_per_thread].copy_from_slice(&values);
                    });
                }
            });
        }

        for i in 0..total {
            assert_eq!(data[i], i as f64);
        }
    }

    #[test]
    fn test_per_worker_accumulator() {
        let n_workers = 4;
        let shape = (3, 2);
        let acc = PerWorkerAccumulator::new(n_workers, shape);

        std::thread::scope(|s| {
            for w in 0..n_workers {
                let acc = &acc;
                s.spawn(move || {
                    let mut guard = acc.get_mut(w);
                    *guard += &Array2::from_elem(shape, (w + 1) as f64);
                });
            }
        });

        let result = acc.sum();
        // Sum of 1+2+3+4 = 10, each element should be 10.0
        let expected = Array2::from_elem(shape, 10.0);
        assert_eq!(result, expected);
    }
}
