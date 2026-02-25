#![cfg(feature = "profiling")]

use anyhow::Result;
use ndarray::Array2;
use ndarray_linalg::{InverseInto, SVD};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use schnelfmm::bed::{decode_bed_chunk_into, BedFile, SnpNorm, SubsetSpec};
use std::sync::Mutex;

use schnelfmm::parallel::{
    subset_indices, PerWorkerAccumulator, SnpBlock,
};
use schnelfmm::precompute::precompute;
use schnelfmm::rsvd::qr_q;
use schnelfmm::simulate::{simulate, write_plink, SimConfig};

extern "C" {
    fn openblas_set_num_threads(num_threads: std::ffi::c_int);
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}

/// Evict a file's pages from the Linux page cache.
///
/// Calls fsync to flush dirty pages, then posix_fadvise(POSIX_FADV_DONTNEED)
/// to tell the kernel to drop them. Subsequent mmap reads will hit disk.
fn drop_page_cache(path: &std::path::Path) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    const POSIX_FADV_DONTNEED: i32 = 4;

    let file = std::fs::File::open(path)?;
    file.sync_all()?;
    let ret = unsafe { posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED) };
    if ret != 0 {
        eprintln!(
            "Warning: posix_fadvise(DONTNEED) failed on {}: error {}",
            path.display(),
            ret,
        );
    }
    Ok(())
}

/// Identify the filesystem type for a path (e.g. "ext4", "tmpfs").
/// Falls back to "unknown" if /proc/mounts can't be parsed.
fn fs_type(path: &std::path::Path) -> String {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let path_str = canonical.to_string_lossy();
    let Ok(mounts) = std::fs::read_to_string("/proc/mounts") else {
        return "unknown".to_string();
    };
    // Find the mount with the longest matching prefix
    let mut best = ("", "unknown");
    for line in mounts.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            continue;
        }
        let mount_point = fields[1];
        let fs = fields[2];
        if path_str.starts_with(mount_point) && mount_point.len() > best.0.len() {
            best = (mount_point, fs);
        }
    }
    let label = best.1.to_string();
    if label == "tmpfs" {
        "tmpfs -- IO times will be 0, use a real disk path".to_string()
    } else {
        label
    }
}

// ---------------------------------------------------------------------------
// Instrumentation (Linux only)
// ---------------------------------------------------------------------------

/// POSIX timespec for clock_gettime FFI.
#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

/// CLOCK_THREAD_CPUTIME_ID (Linux value = 3): per-thread CPU clock.
/// Counts only time the calling thread spent on-CPU. Time blocked on
/// page faults (mmap IO), futex waits, channel recv, etc. is excluded.
const CLOCK_THREAD_CPUTIME_ID: i32 = 3;

extern "C" {
    fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
}

/// Read the calling thread's CPU-time clock (nanoseconds).
/// Returns 0 on failure (non-Linux).
fn thread_cpu_nanos() -> u64 {
    let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
    let ret = unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if ret != 0 {
        return 0;
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

// ---------------------------------------------------------------------------
// /proc-based instrumentation (process-wide, for Phase timing)
// ---------------------------------------------------------------------------

fn cpu_times_secs() -> (f64, f64) {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return (0.0, 0.0);
    };
    let fields: Vec<&str> = stat.split_whitespace().collect();
    if fields.len() < 15 {
        return (0.0, 0.0);
    }
    let ticks_per_sec = 100.0_f64; // sysconf(_SC_CLK_TCK) is almost always 100
    let utime = fields[13].parse::<f64>().unwrap_or(0.0) / ticks_per_sec;
    let stime = fields[14].parse::<f64>().unwrap_or(0.0) / ticks_per_sec;
    (utime, stime)
}

fn rss_mb() -> f64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0.0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: f64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

// ---------------------------------------------------------------------------
// Phase timing
// ---------------------------------------------------------------------------

struct PhaseResult {
    name: String,
    wall_secs: f64,
    cpu_secs: f64,
    rss_mb: f64,
}

impl PhaseResult {
    fn util_pct(&self) -> f64 {
        if self.wall_secs < 1e-9 {
            100.0
        } else {
            100.0 * self.cpu_secs / self.wall_secs
        }
    }
}

struct Phase {
    name: String,
    wall_start: Instant,
    cpu_start: (f64, f64),
}

impl Phase {
    fn start(name: &str) -> Self {
        Phase {
            name: name.to_string(),
            wall_start: Instant::now(),
            cpu_start: cpu_times_secs(),
        }
    }

    fn stop(self) -> PhaseResult {
        let wall_secs = self.wall_start.elapsed().as_secs_f64();
        let (u1, s1) = cpu_times_secs();
        let cpu_secs = (u1 - self.cpu_start.0) + (s1 - self.cpu_start.1);
        PhaseResult {
            name: self.name,
            wall_secs,
            cpu_secs,
            rss_mb: rss_mb(),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamStats — per-pass IO/decode/compute breakdown
// ---------------------------------------------------------------------------

struct StreamStats {
    name: String,
    n_workers: usize,
    wall_secs: f64,
    /// IO thread: wall time per chunk (memcpy from mmap, includes page fault stalls)
    io_ns: AtomicU64,
    max_io_ns: AtomicU64,
    /// Worker thread: decode time per chunk (2-bit→f64, now parallelized)
    decode_ns: AtomicU64,
    max_decode_ns: AtomicU64,
    /// Worker thread: compute time per chunk (matmul, etc.)
    compute_ns: AtomicU64,
    max_compute_ns: AtomicU64,
    chunk_count: AtomicU64,
}

impl StreamStats {
    fn new(name: &str, n_workers: usize) -> Self {
        StreamStats {
            name: name.to_string(),
            n_workers,
            wall_secs: 0.0,
            io_ns: AtomicU64::new(0),
            max_io_ns: AtomicU64::new(0),
            decode_ns: AtomicU64::new(0),
            max_decode_ns: AtomicU64::new(0),
            compute_ns: AtomicU64::new(0),
            max_compute_ns: AtomicU64::new(0),
            chunk_count: AtomicU64::new(0),
        }
    }

    fn chunks(&self) -> u64 {
        self.chunk_count.load(Ordering::Relaxed)
    }

    fn io_secs(&self) -> f64 {
        self.io_ns.load(Ordering::Relaxed) as f64 / 1e9
    }

    fn decode_secs(&self) -> f64 {
        self.decode_ns.load(Ordering::Relaxed) as f64 / 1e9
    }

    fn compute_secs(&self) -> f64 {
        self.compute_ns.load(Ordering::Relaxed) as f64 / 1e9
    }

    fn max_io_ms(&self) -> f64 {
        self.max_io_ns.load(Ordering::Relaxed) as f64 / 1e6
    }

    fn max_decode_ms(&self) -> f64 {
        self.max_decode_ns.load(Ordering::Relaxed) as f64 / 1e6
    }

    fn max_compute_ms(&self) -> f64 {
        self.max_compute_ns.load(Ordering::Relaxed) as f64 / 1e6
    }

    fn avg_io_ms(&self) -> f64 {
        let c = self.chunks();
        if c == 0 { return 0.0; }
        self.io_secs() * 1000.0 / c as f64
    }

    fn avg_decode_ms(&self) -> f64 {
        let c = self.chunks();
        if c == 0 { return 0.0; }
        self.decode_secs() * 1000.0 / c as f64
    }

    fn avg_compute_ms(&self) -> f64 {
        let c = self.chunks();
        if c == 0 { return 0.0; }
        self.compute_secs() * 1000.0 / c as f64
    }

    fn idle_pct(&self) -> f64 {
        if self.wall_secs < 1e-9 {
            return 0.0;
        }
        // IO thread is serial. Workers do decode+compute in parallel.
        let io_total = self.io_secs();
        let worker_total = (self.decode_secs() + self.compute_secs()) / self.n_workers as f64;
        let busy = io_total.max(worker_total);
        let idle = (self.wall_secs - busy).max(0.0);
        100.0 * idle / self.wall_secs
    }

    fn add_io_ns(&self, ns: u64) {
        self.io_ns.fetch_add(ns, Ordering::Relaxed);
        self.max_io_ns.fetch_max(ns, Ordering::Relaxed);
    }

    fn add_decode_ns(&self, ns: u64) {
        self.decode_ns.fetch_add(ns, Ordering::Relaxed);
        self.max_decode_ns.fetch_max(ns, Ordering::Relaxed);
    }

    fn add_compute_ns(&self, ns: u64) {
        self.compute_ns.fetch_add(ns, Ordering::Relaxed);
        self.max_compute_ns.fetch_max(ns, Ordering::Relaxed);
    }

    fn inc_chunks(&self) {
        self.chunk_count.fetch_add(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// profiled_stream — like parallel_stream but with per-phase timing
// ---------------------------------------------------------------------------

/// Copy raw packed bytes for `chunk_indices` from mmap into `dst`.
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

fn profiled_stream<F>(
    bed: &BedFile,
    subset: &SubsetSpec,
    chunk_size: usize,
    n_workers: usize,
    norm: SnpNorm,
    stats: &StreamStats,
    process_fn: F,
) where
    F: Fn(usize, &SnpBlock) + Send + Sync,
{
    let n_workers = n_workers.max(1);
    let indices = subset_indices(subset, bed.n_snps);
    let n_samples = bed.n_samples;
    let bps = bed.bytes_per_snp();
    let mmap_data = bed.mmap_data();

    let pool_size = n_workers + 1;
    let raw_buf_size = chunk_size * bps;
    let local_indices: Vec<usize> = (0..chunk_size).collect();

    let (free_tx, free_rx) = crossbeam_channel::bounded::<SnpBlock>(pool_size);
    let (filled_tx, filled_rx) = crossbeam_channel::bounded::<SnpBlock>(pool_size);

    for _ in 0..pool_size {
        free_tx
            .send(SnpBlock {
                data: Array2::<f64>::zeros((n_samples, chunk_size)),
                n_cols: 0,
                seq: 0,
                raw: vec![0u8; raw_buf_size],
            })
            .unwrap();
    }

    let wall_start = Instant::now();

    std::thread::scope(|s| {
        // IO thread: copies raw bytes from mmap sequentially.
        // Timed with wall-vs-CPU to separate page fault stalls from memcpy.
        s.spawn(|| {
            for (seq, chunk_indices) in indices.chunks(chunk_size).enumerate() {
                let mut block = free_rx.recv().unwrap();
                let n_cols = chunk_indices.len();

                let wall_t0 = Instant::now();
                let cpu_t0 = thread_cpu_nanos();

                copy_raw_chunk(mmap_data, bps, chunk_indices, &mut block.raw);

                let cpu_elapsed = thread_cpu_nanos() - cpu_t0;
                let wall_elapsed = wall_t0.elapsed().as_nanos() as u64;
                // IO = total wall time of memcpy (includes page fault stalls).
                // For the summary table we report total IO thread wall time.
                // The page-fault portion is wall - CPU.
                stats.add_io_ns(wall_elapsed);
                // Store page-fault stall separately for potential future use
                let _page_fault_ns = wall_elapsed.saturating_sub(cpu_elapsed);

                block.n_cols = n_cols;
                block.seq = seq;
                filled_tx.send(block).unwrap();
            }
            drop(filled_tx);
        });

        // Worker threads: decode raw → f64, time decode and compute separately
        for worker_id in 0..n_workers {
            let filled_rx = filled_rx.clone();
            let free_tx = free_tx.clone();
            let process_fn = &process_fn;
            let local_indices = &local_indices;
            s.spawn(move || {
                while let Ok(mut block) = filled_rx.recv() {
                    // Decode: raw packed bytes → centered/scaled f64 matrix
                    let t_dec = Instant::now();
                    let n_cols = block.n_cols;
                    let out_view = block.data.slice_mut(ndarray::s![.., ..n_cols]);
                    decode_bed_chunk_into(
                        &block.raw,
                        bps,
                        n_samples,
                        &local_indices[..n_cols],
                        out_view,
                        norm,
                        None,
                    );
                    stats.add_decode_ns(t_dec.elapsed().as_nanos() as u64);

                    // Compute: user closure (matmul, etc.)
                    let t_comp = Instant::now();
                    process_fn(worker_id, &block);
                    stats.add_compute_ns(t_comp.elapsed().as_nanos() as u64);

                    stats.inc_chunks();
                    let _ = free_tx.send(block);
                }
            });
        }
    });

    // SAFETY: we're the only writer of wall_secs and all threads have joined
    let wall_secs = wall_start.elapsed().as_secs_f64();
    let stats_ptr = stats as *const StreamStats as *mut StreamStats;
    unsafe { (*stats_ptr).wall_secs = wall_secs; }
}

// ---------------------------------------------------------------------------
// Helpers duplicated from testing.rs (kept minimal)
// ---------------------------------------------------------------------------

fn t_test(coef: f64, sigma2: f64, ctc_inv_jj: f64, df: f64) -> (f64, f64) {
    let se = (sigma2 * ctc_inv_jj).sqrt();
    if se < 1e-300 || !se.is_finite() {
        return (0.0, 1.0);
    }
    let t = coef / se;
    if !t.is_finite() {
        return (0.0, 1.0);
    }
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let p_val = 2.0 * t_dist.cdf(-t.abs());
    (t, p_val)
}

fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn safe_inv(a: &Array2<f64>) -> Result<Array2<f64>> {
    Ok(a.clone().inv_into()?)
}

// ---------------------------------------------------------------------------
// Profile test
// ---------------------------------------------------------------------------

/// Runs the full LFMM2 pipeline with timing instrumentation.
///
/// Uses default profiling parameters (n=500, p=50_000, K=5, d=2).
/// Gated behind the `profiling` feature and marked `#[ignore]` so it
/// doesn't run in normal `cargo test`; invoke with:
///
///   cargo test --features profiling --test profile -- --ignored --nocapture
#[test]
#[ignore]
fn profile_pipeline() -> Result<()> {
    // Default parameters (matching the former CLI defaults)
    let n = 500_usize;
    let p = 50_000_usize;
    let k = 5_usize;
    let d = 2_usize;
    let chunk_size = 10_000_usize;
    let n_workers = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1))
        .unwrap_or(1)
        .max(1);
    let oversampling = 10_usize;
    let l = k + oversampling;
    let n_power_iter = 2_usize;
    let seed = 42_u64;
    let lambda = 1e-5_f64;

    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("MKL_NUM_THREADS", "1");
    unsafe { openblas_set_num_threads(1); }

    eprintln!(
        "LFMM2 Profile: n={}, p={}, K={}, d={}, threads={}, chunk_size={}",
        n, p, k, d, n_workers, chunk_size,
    );
    eprintln!(
        "  oversampling={}, power_iter={}, seed={}, lambda={:.1e}",
        oversampling, n_power_iter, seed, lambda,
    );
    eprintln!("{}", "=".repeat(74));

    let mut phases: Vec<PhaseResult> = Vec::new();
    let mut stream_stats: Vec<StreamStats> = Vec::new();

    // -----------------------------------------------------------------------
    // Phase 1a: Simulate
    // -----------------------------------------------------------------------
    let ph = Phase::start("Data: simulate");
    let sim = simulate(&SimConfig {
        n_samples: n,
        n_snps: p,
        n_causal: (p as f64 * 0.01).ceil() as usize,
        k,
        d,
        effect_size: 0.5,
        latent_scale: 1.0,
        noise_std: 1.0,
        covariate_r2: 0.0,
        seed,
    });
    phases.push(ph.stop());

    // -----------------------------------------------------------------------
    // Phase 1b: Write PLINK + drop page cache + open BED
    // -----------------------------------------------------------------------
    let ph = Phase::start("Data: write PLINK");
    let tmp_dir = tempfile::Builder::new()
        .prefix(".lfmm2_profile_")
        .tempdir_in(".")?;
    eprintln!("  tmpdir: {} ({})",
        tmp_dir.path().display(),
        fs_type(tmp_dir.path()),
    );
    write_plink(tmp_dir.path(), "profile", &sim)?;
    phases.push(ph.stop());

    let ph = Phase::start("Data: drop page cache");
    let bed_path = tmp_dir.path().join("profile.bed");
    drop_page_cache(&bed_path)?;
    phases.push(ph.stop());

    let ph = Phase::start("Data: open BED (mmap)");
    let bed = BedFile::open(&bed_path)?;
    phases.push(ph.stop());

    let subset = SubsetSpec::All;
    let x = &sim.x;

    // -----------------------------------------------------------------------
    // Phase 2: Precompute
    // -----------------------------------------------------------------------
    let ph = Phase::start("Precompute: SVD(X) + build M");
    let pre = precompute(x, lambda)?;
    phases.push(ph.stop());

    // -----------------------------------------------------------------------
    // Phase 3: RSVD factor estimation (replicated from rsvd.rs)
    // -----------------------------------------------------------------------
    let p_est = bed.subset_snp_count(&subset);

    // 3a: Generate Omega, Mt_omega
    let ph = Phase::start("RSVD: generate Omega + Mt_omega");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    let omega = Array2::from_shape_fn((n, l), |_| normal_dist.sample(&mut rng));
    let mt_omega = pre.m.t().dot(&omega);
    phases.push(ph.stop());

    // 3b: Sketch pass
    let mut z = Array2::<f64>::zeros((p_est, l));
    {
        let ss = StreamStats::new("Sketch", n_workers);
        let z_mutex = Mutex::new(&mut z);
        profiled_stream(&bed, &subset, chunk_size, n_workers, SnpNorm::Eigenstrat, &ss, |_wid, block| {
            let chunk = block.data.slice(ndarray::s![.., ..block.n_cols]);
            let z_block = chunk.t().dot(&mt_omega);
            let start = block.seq * chunk_size;
            z_mutex.lock().unwrap()
                .slice_mut(ndarray::s![start..start + z_block.nrows(), ..])
                .assign(&z_block);
        });
        phases.push(PhaseResult {
            name: "RSVD: sketch pass".to_string(),
            wall_secs: ss.wall_secs,
            cpu_secs: 0.0,
            rss_mb: rss_mb(),
        });
        stream_stats.push(ss);
    }

    // 3c: QR(Z)
    let ph = Phase::start("RSVD: QR(Z)");
    let mut q_z = qr_q(&z);
    phases.push(ph.stop());

    // Power iterations
    for iter in 0..n_power_iter {
        // Forward pass: A @ Q_z = M @ Y_est @ Q_z
        let a_qz;
        {
            let label = format!("Power {} fwd", iter + 1);
            let ss = StreamStats::new(&label, n_workers);
            let acc = PerWorkerAccumulator::new(n_workers, (n, l));
            profiled_stream(&bed, &subset, chunk_size, n_workers, SnpNorm::Eigenstrat, &ss, |wid, block| {
                let chunk = block.data.slice(ndarray::s![.., ..block.n_cols]);
                let offset = block.seq * chunk_size;
                let q_z_block = q_z.slice(ndarray::s![offset..offset + block.n_cols, ..]);
                let y_qz = chunk.dot(&q_z_block);
                let partial = pre.m.dot(&y_qz);
                *acc.get_mut(wid) += &partial;
            });
            a_qz = acc.sum();
            phases.push(PhaseResult {
                name: format!("RSVD: power {} fwd", iter + 1),
                wall_secs: ss.wall_secs,
                cpu_secs: 0.0,
                rss_mb: rss_mb(),
            });
            stream_stats.push(ss);
        }

        // QR(AQz)
        let ph = Phase::start(&format!("RSVD: QR(AQz) iter {}", iter + 1));
        let q_aqz = qr_q(&a_qz);
        phases.push(ph.stop());

        // Mt_q = M^T @ Q_aqz
        let ph = Phase::start(&format!("RSVD: Mt_q iter {}", iter + 1));
        let mt_q = pre.m.t().dot(&q_aqz);
        phases.push(ph.stop());

        // Backward pass
        z = Array2::<f64>::zeros((p_est, l));
        {
            let label = format!("Power {} bwd", iter + 1);
            let ss = StreamStats::new(&label, n_workers);
            let z_mutex = Mutex::new(&mut z);
            profiled_stream(&bed, &subset, chunk_size, n_workers, SnpNorm::Eigenstrat, &ss, |_wid, block| {
                let chunk = block.data.slice(ndarray::s![.., ..block.n_cols]);
                let z_block = chunk.t().dot(&mt_q);
                let start = block.seq * chunk_size;
                z_mutex.lock().unwrap()
                    .slice_mut(ndarray::s![start..start + z_block.nrows(), ..])
                    .assign(&z_block);
            });
            phases.push(PhaseResult {
                name: format!("RSVD: power {} bwd", iter + 1),
                wall_secs: ss.wall_secs,
                cpu_secs: 0.0,
                rss_mb: rss_mb(),
            });
            stream_stats.push(ss);
        }

        // QR(Z)
        let ph = Phase::start(&format!("RSVD: QR(Z) iter {}", iter + 1));
        q_z = qr_q(&z);
        phases.push(ph.stop());
    }

    // Final project pass
    let b_svd;
    {
        let ss = StreamStats::new("Final project", n_workers);
        let acc = PerWorkerAccumulator::new(n_workers, (n, l));
        profiled_stream(&bed, &subset, chunk_size, n_workers, SnpNorm::Eigenstrat, &ss, |wid, block| {
            let chunk = block.data.slice(ndarray::s![.., ..block.n_cols]);
            let offset = block.seq * chunk_size;
            let q_z_block = q_z.slice(ndarray::s![offset..offset + block.n_cols, ..]);
            let y_qz = chunk.dot(&q_z_block);
            let partial = pre.m.dot(&y_qz);
            *acc.get_mut(wid) += &partial;
        });
        b_svd = acc.sum();
        phases.push(PhaseResult {
            name: "RSVD: final project".to_string(),
            wall_secs: ss.wall_secs,
            cpu_secs: 0.0,
            rss_mb: rss_mb(),
        });
        stream_stats.push(ss);
    }

    // Small SVD + recover U_hat
    let ph = Phase::start("RSVD: small SVD + recover");
    let (u_opt, _s, _vt_opt) = b_svd.svd(true, false)?;
    let u_small = u_opt.expect("SVD should return U");
    let u_small_k = u_small.slice(ndarray::s![.., ..k]).to_owned();
    let mut dlam_inv_u = u_small_k.clone();
    for i in 0..n {
        dlam_inv_u
            .row_mut(i)
            .mapv_inplace(|v| v * pre.d_lambda_inv[i]);
    }
    let u_hat = pre.q_full.dot(&dlam_inv_u);
    phases.push(ph.stop());

    // -----------------------------------------------------------------------
    // Phase 4: Association testing (replicated from testing.rs)
    // -----------------------------------------------------------------------

    // 4a: Precompute C, H, etc.
    let ph = Phase::start("Assoc: precompute C,H");
    let df = (n - 1 - d - k) as f64;

    let utu = u_hat.t().dot(&u_hat);
    let utu_inv = safe_inv(&utu)?;
    let p_u = u_hat.dot(&utu_inv).dot(&u_hat.t());

    let xtr = pre.ridge_inv.dot(&x.t());

    let mut i_minus_pu = Array2::<f64>::eye(n);
    i_minus_pu -= &p_u;

    let c_cols = 1 + d + k;
    let mut c = Array2::<f64>::zeros((n, c_cols));
    c.column_mut(0).fill(1.0); // intercept
    c.slice_mut(ndarray::s![.., 1..1 + d]).assign(x);
    c.slice_mut(ndarray::s![.., 1 + d..]).assign(&u_hat);

    let ctc = c.t().dot(&c);
    let ctc_inv = safe_inv(&ctc)?;
    let h = ctc_inv.dot(&c.t());
    let ctc_inv_diag: Vec<f64> = (0..d).map(|j| ctc_inv[(1 + j, 1 + j)]).collect();
    phases.push(ph.stop());

    // 4b: Streaming association test pass
    let mut effect_sizes = Array2::<f64>::zeros((p, d));
    let mut t_stats = Array2::<f64>::zeros((p, d));
    {
        let ss = StreamStats::new("Assoc test", n_workers);
        let mtx_effects = Mutex::new(&mut effect_sizes);
        let mtx_tstats = Mutex::new(&mut t_stats);
        profiled_stream(&bed, &SubsetSpec::All, chunk_size, n_workers, SnpNorm::Eigenstrat, &ss, |_wid, block| {
            let chunk = block.data.slice(ndarray::s![.., ..block.n_cols]);
            let chunk_cols = block.n_cols;
            let start = block.seq * chunk_size;

            // Step 3: B
            let residual = i_minus_pu.dot(&chunk);
            let b_chunk = xtr.dot(&residual);
            let b_chunk_t = b_chunk.t().to_owned();

            // Step 4: OLS + t-test
            let coefs = h.dot(&chunk);
            let fitted = c.dot(&coefs);
            let residuals = &chunk - &fitted;

            let mut local_tstats = Array2::<f64>::zeros((chunk_cols, d));
            for col_in_chunk in 0..chunk_cols {
                let res_col = residuals.column(col_in_chunk);
                let rss: f64 = res_col.dot(&res_col);
                let sigma2 = rss / df;
                for j in 0..d {
                    let (t, _) = t_test(coefs[(1 + j, col_in_chunk)], sigma2, ctc_inv_diag[j], df);
                    local_tstats[(col_in_chunk, j)] = t;
                }
            }

            mtx_effects.lock().unwrap()
                .slice_mut(ndarray::s![start..start + chunk_cols, ..])
                .assign(&b_chunk_t);
            mtx_tstats.lock().unwrap()
                .slice_mut(ndarray::s![start..start + chunk_cols, ..])
                .assign(&local_tstats);
        });
        phases.push(PhaseResult {
            name: "Assoc: streaming pass".to_string(),
            wall_secs: ss.wall_secs,
            cpu_secs: 0.0,
            rss_mb: rss_mb(),
        });
        stream_stats.push(ss);
    }

    // 4c: GIF calibration
    let ph = Phase::start("Assoc: GIF calibration");
    let normal = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
    let mut avg_gif = 0.0;

    for j in 0..d {
        let t_col = t_stats.column(j);
        let mut z_sq: Vec<f64> = t_col.iter().map(|&t| t * t).filter(|v| v.is_finite()).collect();
        z_sq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_z_sq = median_sorted(&z_sq);
        let gif = if median_z_sq < 1e-10 { 1.0 } else { median_z_sq / 0.4549 };
        avg_gif += gif;
        let gif_sqrt = gif.sqrt();
        for i in 0..p {
            let z = t_stats[(i, j)];
            let z_cal = z / gif_sqrt;
            let _ = 2.0 * normal.cdf(-z_cal.abs());
        }
    }
    avg_gif /= d as f64;
    phases.push(ph.stop());

    // -----------------------------------------------------------------------
    // Print summary table
    // -----------------------------------------------------------------------
    eprintln!();
    eprintln!(
        "{:<38} {:>8} {:>8} {:>6} {:>9}",
        "Phase", "Wall(s)", "CPU(s)", "Util%", "RSS(MB)"
    );
    eprintln!("{}", "-".repeat(74));

    let mut total_wall = 0.0;
    let mut total_cpu = 0.0;
    let mut max_rss = 0.0_f64;

    for ph in &phases {
        let util = ph.util_pct();
        eprintln!(
            "{:<38} {:>8.2} {:>8.2} {:>5.0}% {:>8.0}",
            ph.name, ph.wall_secs, ph.cpu_secs, util, ph.rss_mb,
        );
        total_wall += ph.wall_secs;
        total_cpu += ph.cpu_secs;
        max_rss = max_rss.max(ph.rss_mb);
    }

    eprintln!("{}", "-".repeat(74));
    let total_util = if total_wall > 1e-9 {
        100.0 * total_cpu / total_wall
    } else {
        100.0
    };
    eprintln!(
        "{:<38} {:>8.2} {:>8.2} {:>5.0}% {:>8.0}",
        "Total", total_wall, total_cpu, total_util, max_rss,
    );

    // -----------------------------------------------------------------------
    // Streaming pass details
    // -----------------------------------------------------------------------
    if !stream_stats.is_empty() {
        eprintln!();
        eprintln!("Streaming pass details:");
        eprintln!(
            "  {:<20} {:>7} {:>7} {:>10} {:>11} {:>8} {:>6}",
            "Pass", "Chunks", "IO(s)", "Decode(s)", "Compute(s)", "Wall(s)", "Idle%"
        );
        for ss in &stream_stats {
            eprintln!(
                "  {:<20} {:>7} {:>7.3} {:>10.3} {:>11.3} {:>8.3} {:>5.0}%",
                ss.name,
                ss.chunks(),
                ss.io_secs(),
                ss.decode_secs(),
                ss.compute_secs(),
                ss.wall_secs,
                ss.idle_pct(),
            );
        }

        eprintln!();
        eprintln!("  Per-chunk averages:");
        eprintln!(
            "  {:<20} {:>7} {:>10} {:>11} {:>9} {:>13} {:>14}",
            "Pass", "IO(ms)", "Decode(ms)", "Compute(ms)", "MaxIO(ms)", "MaxDecode(ms)", "MaxCompute(ms)"
        );
        for ss in &stream_stats {
            eprintln!(
                "  {:<20} {:>7.1} {:>10.1} {:>11.1} {:>9.1} {:>13.1} {:>14.1}",
                ss.name,
                ss.avg_io_ms(),
                ss.avg_decode_ms(),
                ss.avg_compute_ms(),
                ss.max_io_ms(),
                ss.max_decode_ms(),
                ss.max_compute_ms(),
            );
        }
    }

    eprintln!();
    eprintln!("GIF: {:.4}", avg_gif);

    // Smoke-check: GIF should be in a reasonable range
    assert!(avg_gif > 0.0, "GIF should be positive");

    Ok(())
}
