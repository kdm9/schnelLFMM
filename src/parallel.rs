use ndarray::Array2;
use std::cell::UnsafeCell;

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

/// Allows multiple threads to write to disjoint row ranges of an Array2 without locking.
///
/// Safety invariant: each `seq` value maps to a unique, non-overlapping row range.
/// This is guaranteed by construction — seq values are assigned sequentially by a single
/// IO thread and each maps to `[seq * chunk_size .. seq * chunk_size + n_cols]`.
pub struct DisjointRowWriter {
    ptr: *mut f64,
    n_rows: usize,
    n_cols: usize,
}

unsafe impl Send for DisjointRowWriter {}
unsafe impl Sync for DisjointRowWriter {}

impl DisjointRowWriter {
    /// Create a writer over a mutable Array2 in standard (row-major) layout.
    ///
    /// The array must remain valid and not be accessed mutably elsewhere
    /// for the lifetime of this writer.
    pub fn new(arr: &mut Array2<f64>) -> Self {
        assert!(arr.is_standard_layout(), "Array must be row-major (standard layout)");
        DisjointRowWriter {
            ptr: arr.as_mut_ptr(),
            n_rows: arr.nrows(),
            n_cols: arr.ncols(),
        }
    }

    /// Write `src` into rows `[row_start .. row_start + src.nrows()]`.
    ///
    /// # Safety
    /// Caller must ensure no two threads write to overlapping row ranges.
    pub unsafe fn write_rows(&self, row_start: usize, src: &Array2<f64>) {
        let src_rows = src.nrows();
        let src_cols = src.ncols();
        assert!(
            row_start + src_rows <= self.n_rows,
            "write_rows: row_start={} + src_rows={} > n_rows={}",
            row_start,
            src_rows,
            self.n_rows
        );
        assert_eq!(
            src_cols, self.n_cols,
            "write_rows: src_cols={} != n_cols={}",
            src_cols, self.n_cols
        );

        if src.is_standard_layout() {
            // Fast path: bulk copy
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                self.ptr.add(row_start * self.n_cols),
                src_rows * src_cols,
            );
        } else {
            // Slow path: row-by-row
            for r in 0..src_rows {
                for c in 0..src_cols {
                    let dst = self.ptr.add((row_start + r) * self.n_cols + c);
                    *dst = src[(r, c)];
                }
            }
        }
    }
}

/// Lock-free per-worker accumulator for forward passes that sum partial results.
///
/// Each worker thread writes exclusively to its own buffer (indexed by worker_id).
/// After all workers finish, call `.sum()` to merge all buffers.
///
/// Safety invariant: each worker_id must be used by exactly one thread at a time.
/// This is guaranteed by `parallel_stream`, which assigns unique indices to workers.
pub struct PerWorkerAccumulator {
    buffers: Vec<UnsafeCell<Array2<f64>>>,
}

unsafe impl Sync for PerWorkerAccumulator {}

impl PerWorkerAccumulator {
    /// Create `n_workers` zero-initialized accumulators of the given shape.
    pub fn new(n_workers: usize, shape: (usize, usize)) -> Self {
        let buffers = (0..n_workers)
            .map(|_| UnsafeCell::new(Array2::<f64>::zeros(shape)))
            .collect();
        PerWorkerAccumulator { buffers }
    }

    /// Get a mutable reference to the buffer for `worker_id`.
    ///
    /// # Safety
    /// Caller must ensure no two threads access the same `worker_id` concurrently.
    pub unsafe fn get_mut(&self, worker_id: usize) -> &mut Array2<f64> {
        &mut *self.buffers[worker_id].get()
    }

    /// Consume the accumulator and return the element-wise sum of all buffers.
    pub fn sum(self) -> Array2<f64> {
        let mut buffers: Vec<Array2<f64>> = self
            .buffers
            .into_iter()
            .map(|cell| cell.into_inner())
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
