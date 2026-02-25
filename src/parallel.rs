use ndarray::Array2;
use std::sync::Mutex;

use crate::bed::{decode_bed_chunk_into, BedFile, SnpNorm, SubsetSpec};

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
///   This keeps disk reads in order (important for spinning disks and readahead).
/// - **Worker threads**: decode the raw 2-bit→f64 data, then call `process_fn`.
///   Decode is CPU-bound and benefits from parallelization across workers.
///
/// Uses `std::thread::scope` so that `process_fn` can borrow from the caller's stack
/// without requiring `'static` bounds.
pub fn parallel_stream<F>(
    bed: &BedFile,
    subset: &SubsetSpec,
    chunk_size: usize,
    n_workers: usize,
    norm: SnpNorm,
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
    let mmap_data = &bed.mmap[3..]; // skip 3-byte magic header

    let pool_size = n_workers + 1;
    let raw_buf_size = chunk_size * bps;

    // Pre-compute local indices [0, 1, ..., chunk_size-1] for decode_bed_chunk_into.
    // After the IO thread packs raw bytes contiguously, SNP i is at offset i*bps.
    let local_indices: Vec<usize> = (0..chunk_size).collect();

    // Create channels
    let (free_tx, free_rx) = crossbeam_channel::bounded::<SnpBlock>(pool_size);
    let (filled_tx, filled_rx) = crossbeam_channel::bounded::<SnpBlock>(pool_size);

    // Seed the free pool with pre-allocated blocks
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

    std::thread::scope(|s| {
        // IO thread: copies raw bytes from mmap sequentially, sends to workers
        s.spawn(|| {
            for (seq, chunk_indices) in indices.chunks(chunk_size).enumerate() {
                let mut block = free_rx.recv().unwrap();
                let n_cols = chunk_indices.len();

                copy_raw_chunk(mmap_data, bps, chunk_indices, &mut block.raw);

                block.n_cols = n_cols;
                block.seq = seq;
                filled_tx.send(block).unwrap();
            }
            // Drop sender to signal workers that no more blocks are coming
            drop(filled_tx);
        });

        // Worker threads: decode raw bytes → f64, then call process_fn
        for worker_id in 0..n_workers {
            let filled_rx = filled_rx.clone();
            let free_tx = free_tx.clone();
            let process_fn = &process_fn;
            let local_indices = &local_indices;
            s.spawn(move || {
                while let Ok(mut block) = filled_rx.recv() {
                    // Decode: raw packed bytes → centered/scaled f64 matrix
                    let n_cols = block.n_cols;
                    let out_view = block.data.slice_mut(ndarray::s![.., ..n_cols]);
                    decode_bed_chunk_into(
                        &block.raw,
                        bps,
                        n_physical_samples,
                        &local_indices[..n_cols],
                        out_view,
                        norm,
                        sample_keep,
                    );

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
