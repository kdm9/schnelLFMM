use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// PLINK .bed magic bytes: 0x6C, 0x1B, 0x01 (SNP-major mode)
const BED_MAGIC: [u8; 3] = [0x6C, 0x1B, 0x01];

/// 2-bit decode table: PLINK encoding -> genotype value
/// 0b00 = hom ref  -> 0.0
/// 0b01 = missing  -> NaN (sentinel)
/// 0b10 = het      -> 1.0
/// 0b11 = hom alt  -> 2.0
const DECODE_TABLE: [f64; 4] = [0.0, f64::NAN, 1.0, 2.0];

/// SNP normalization mode applied after centering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum SnpNorm {
    /// Center only — subtract mean, no variance scaling.
    CenterOnly,
    /// Eigenstrat — divide by sqrt(2pq) for approximately unit variance under HWE.
    #[default]
    Eigenstrat,
    /// HWE — multiply by sqrt(2pq) so Var(g) = 2pq exactly.
    Hwe,
}

#[derive(Debug, Clone)]
pub struct BimRecord {
    pub chrom: String,
    pub snp_id: String,
    pub cm: f64,
    pub pos: u64,
    pub allele1: String,
    pub allele2: String,
}

#[derive(Debug, Clone)]
pub struct FamRecord {
    pub fid: String,
    pub iid: String,
    pub father: String,
    pub mother: String,
    pub sex: u8,
    pub pheno: String,
}

/// Subset specification for estimation
#[derive(Debug, Clone)]
pub enum SubsetSpec {
    /// Use all SNPs
    All,
    /// Random thinning at given rate (0.0, 1.0]
    Rate(f64),
    /// Specific SNP indices
    Indices(Vec<usize>),
}

/// On-disk PLINK .bed/.bim/.fam triplet
pub struct BedFile {
    pub bed_path: PathBuf,
    pub n_samples: usize,
    pub n_snps: usize,
    pub bim_records: Vec<BimRecord>,
    pub fam_records: Vec<FamRecord>,
    pub(crate) mmap: Mmap,
    /// Original .fam sample count (for reading packed bytes from .bed).
    /// After `subset_samples()`, `n_samples` shrinks but the on-disk encoding
    /// is still based on `n_physical_samples`.
    pub n_physical_samples: usize,
    /// When set, only these physical-sample indices are decoded/output.
    /// Indices are into the original .fam order (before subsetting).
    pub sample_keep: Option<Vec<usize>>,
}

impl BedFile {
    /// Open a PLINK .bed file and parse companion .bim/.fam files.
    pub fn open(bed_path: impl AsRef<Path>) -> Result<Self> {
        let bed_path = bed_path.as_ref().to_path_buf();

        // Derive .bim and .fam paths
        let stem = bed_path.with_extension("");
        let bim_path = stem.with_extension("bim");
        let fam_path = stem.with_extension("fam");

        let fam_records = parse_fam(&fam_path)
            .with_context(|| format!("Failed to parse {}", fam_path.display()))?;
        let bim_records = parse_bim(&bim_path)
            .with_context(|| format!("Failed to parse {}", bim_path.display()))?;

        let n_samples = fam_records.len();
        let n_snps = bim_records.len();

        // Memory-map the .bed file
        let file = File::open(&bed_path)
            .with_context(|| format!("Failed to open {}", bed_path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Hint to the OS that we'll read sequentially (improves readahead)
        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Sequential)
            .with_context(|| format!("madvise(SEQUENTIAL) failed for {}", bed_path.display()))?;

        // Validate magic bytes
        if mmap.len() < 3 || mmap[0..3] != BED_MAGIC {
            bail!(
                "Invalid .bed file (bad magic bytes): {}",
                bed_path.display()
            );
        }

        // Validate file size
        let bytes_per_snp = n_samples.div_ceil(4);
        let expected_size = 3 + bytes_per_snp * n_snps;
        if mmap.len() != expected_size {
            bail!(
                "BED file size mismatch: expected {} bytes ({} samples × {} SNPs), got {}",
                expected_size,
                n_samples,
                n_snps,
                mmap.len()
            );
        }

        Ok(BedFile {
            bed_path,
            n_physical_samples: n_samples,
            n_samples,
            n_snps,
            bim_records,
            fam_records,
            mmap,
            sample_keep: None,
        })
    }

    /// Raw mmap bytes after the 3-byte magic header.
    pub fn mmap_data(&self) -> &[u8] {
        &self.mmap[3..]
    }

    /// Bytes per SNP in the .bed file (ceil(n_physical_samples / 4)).
    /// Uses the physical (on-disk) sample count, not the post-subsetting count.
    pub fn bytes_per_snp(&self) -> usize {
        self.n_physical_samples.div_ceil(4)
    }

    /// Subset this BedFile to keep only the given physical-sample indices.
    ///
    /// `indices` are positions in the original .fam order. After calling this:
    /// - `n_samples` = `indices.len()`
    /// - `fam_records` is reordered to match `indices`
    /// - `sample_keep` is set for decode functions
    /// - `n_physical_samples` and `bytes_per_snp()` remain unchanged
    pub fn subset_samples(&mut self, indices: Vec<usize>) {
        self.fam_records = indices.iter().map(|&i| self.fam_records[i].clone()).collect();
        self.n_samples = indices.len();
        self.sample_keep = Some(indices);
    }

    /// Total number of SNPs that will be yielded by this subset
    pub fn subset_snp_count(&self, subset: &SubsetSpec) -> usize {
        match subset {
            SubsetSpec::All => self.n_snps,
            SubsetSpec::Rate(rate) => {
                let step = (1.0 / rate).ceil() as usize;
                self.n_snps.div_ceil(step)
            }
            SubsetSpec::Indices(indices) => indices.len(),
        }
    }
}

/// Decode a chunk of SNPs from .bed format into a centered f64 matrix.
/// Missing values (0b01) are imputed to the per-SNP mean genotype, then centered.
///
/// - `n_physical_samples`: the on-disk sample count (determines bytes per SNP)
/// - `n_output_samples`: rows in the output matrix (equals `n_physical_samples` when
///   `sample_keep` is `None`, or `sample_keep.len()` otherwise)
/// - `sample_keep`: if `Some`, only these physical-sample indices are decoded
pub fn decode_bed_chunk(
    packed: &[u8],
    n_physical_samples: usize,
    n_output_samples: usize,
    chunk_size: usize,
    norm: SnpNorm,
    sample_keep: Option<&[usize]>,
) -> Array2<f64> {
    let bytes_per_snp = n_physical_samples.div_ceil(4);
    let mut out = Array2::<f64>::zeros((n_output_samples, chunk_size));

    for snp in 0..chunk_size {
        let snp_bytes = &packed[snp * bytes_per_snp..(snp + 1) * bytes_per_snp];
        decode_single_snp(snp_bytes, n_physical_samples, out.column_mut(snp), norm, sample_keep);
    }
    out
}

/// Decode specific SNPs from mmap data into a pre-allocated output matrix.
///
/// - `mmap_data`: mmap bytes after the 3-byte header
/// - `bps`: bytes per SNP (based on physical sample count)
/// - `n_physical_samples`: on-disk sample count
/// - `snp_indices`: which SNPs to decode (column indices in the .bed file)
/// - `out`: pre-allocated output (n_output_samples × snp_indices.len())
/// - `sample_keep`: if `Some`, only these physical-sample indices are decoded
pub fn decode_bed_chunk_into(
    mmap_data: &[u8],
    bps: usize,
    n_physical_samples: usize,
    snp_indices: &[usize],
    mut out: ndarray::ArrayViewMut2<f64>,
    norm: SnpNorm,
    sample_keep: Option<&[usize]>,
) {
    for (col, &snp_idx) in snp_indices.iter().enumerate() {
        let snp_bytes = &mmap_data[snp_idx * bps..(snp_idx + 1) * bps];
        decode_single_snp(snp_bytes, n_physical_samples, out.column_mut(col), norm, sample_keep);
    }
}

/// Decode a single SNP column, impute missing to mean, center, and optionally scale.
///
/// After decoding 2-bit genotypes to {0, 1, 2}, this function:
/// 1. Computes the per-SNP mean (excluding missing values)
/// 2. Imputes missing values to the mean (centered = 0)
/// 3. Centers all values by subtracting the mean
/// 4. Applies normalization according to `norm`:
///    - `CenterOnly`: no scaling (scale = 1.0)
///    - `Eigenstrat`: divides by sqrt(2pq) for approx unit variance under HWE
///    - `Hwe`: multiplies by sqrt(2pq) so Var(g) = 2pq exactly
///
/// Monomorphic SNPs (p=0 or p=1) have zero variance and are left as all zeros.
///
/// When `sample_keep` is `Some`, only those physical-sample indices are decoded
/// into `col` (which has length `sample_keep.len()`). Mean/variance statistics
/// are computed over the kept samples only.
pub(crate) fn decode_single_snp(
    snp_bytes: &[u8],
    n_physical_samples: usize,
    mut col: ndarray::ArrayViewMut1<f64>,
    norm: SnpNorm,
    sample_keep: Option<&[usize]>,
) {
    if let Some(keep) = sample_keep {
        // Subsetting path: decode only kept samples
        let mut sum = 0.0f64;
        let mut n_valid = 0u32;

        // Pass 1: decode kept samples and compute mean
        for (out_idx, &phys_idx) in keep.iter().enumerate() {
            debug_assert!(phys_idx < n_physical_samples);
            let byte = snp_bytes[phys_idx / 4];
            let code = (byte >> (2 * (phys_idx % 4))) & 0x03;
            let val = DECODE_TABLE[code as usize];
            col[out_idx] = val;
            if !val.is_nan() {
                sum += val;
                n_valid += 1;
            }
        }

        let mean = if n_valid > 0 { sum / n_valid as f64 } else { 0.0 };
        let scale = compute_scale(mean, norm);

        // Pass 2: impute, center, scale
        col.mapv_inplace(|v| if v.is_nan() { 0.0 } else { (v - mean) * scale });
    } else {
        // Original fast path: all samples
        let n_samples = n_physical_samples;
        let mut sum = 0.0f64;
        let mut n_valid = 0u32;

        for sample in 0..n_samples {
            let byte = snp_bytes[sample / 4];
            let code = (byte >> (2 * (sample % 4))) & 0x03;
            let val = DECODE_TABLE[code as usize];
            col[sample] = val;
            if !val.is_nan() {
                sum += val;
                n_valid += 1;
            }
        }

        let mean = if n_valid > 0 { sum / n_valid as f64 } else { 0.0 };
        let scale = compute_scale(mean, norm);

        col.mapv_inplace(|v| if v.is_nan() { 0.0 } else { (v - mean) * scale });
    }
}

/// Compute the normalization scale factor from the SNP mean.
fn compute_scale(mean: f64, norm: SnpNorm) -> f64 {
    let p = mean / 2.0;
    let twopq = 2.0 * p * (1.0 - p);
    match norm {
        SnpNorm::CenterOnly => {
            // Invariant sites have zero variance, set all values to zero (in case all are 2)
            if twopq < 1e-20 { 0.0 } else { 1.0 }
        }
        SnpNorm::Eigenstrat => {
            let denom = twopq.sqrt();
            if denom > 1e-10 { 1.0 / denom } else { 0.0 }
        }
        SnpNorm::Hwe => {
            let s = twopq.sqrt();
            if s < 1e-10 { 0.0 } else { s }
        }
    }
}

/// Write a PLINK .bed file from genotype matrix (n_samples × n_snps, values in {0,1,2}).
pub fn write_bed_file(path: &Path, genotypes: &Array2<u8>) -> Result<()> {
    let n_samples = genotypes.nrows();
    let n_snps = genotypes.ncols();
    let bytes_per_snp = n_samples.div_ceil(4);

    let mut file = File::create(path)?;
    file.write_all(&BED_MAGIC)?;

    // Encode table: genotype value -> 2-bit code
    // 0 -> 0b00, 1 -> 0b10, 2 -> 0b11, missing (255) -> 0b01
    let encode = |g: u8| -> u8 {
        match g {
            0 => 0b00,
            1 => 0b10,
            2 => 0b11,
            _ => 0b01, // missing
        }
    };

    let mut buf = vec![0u8; bytes_per_snp];
    for snp in 0..n_snps {
        buf.fill(0);
        for sample in 0..n_samples {
            let code = encode(genotypes[(sample, snp)]);
            buf[sample / 4] |= code << (2 * (sample % 4));
        }
        file.write_all(&buf)?;
    }

    Ok(())
}

/// Write a .bim file
pub fn write_bim(path: &Path, records: &[BimRecord]) -> Result<()> {
    let mut file = File::create(path)?;
    for r in records {
        writeln!(
            file,
            "{}\t{}\t{}\t{}\t{}\t{}",
            r.chrom, r.snp_id, r.cm, r.pos, r.allele1, r.allele2
        )?;
    }
    Ok(())
}

/// Write a .fam file
pub fn write_fam(path: &Path, records: &[FamRecord]) -> Result<()> {
    let mut file = File::create(path)?;
    for r in records {
        writeln!(
            file,
            "{}\t{}\t{}\t{}\t{}\t{}",
            r.fid, r.iid, r.father, r.mother, r.sex, r.pheno
        )?;
    }
    Ok(())
}

fn parse_fam(path: &Path) -> Result<Vec<FamRecord>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 6 {
            bail!("FAM line {} has {} fields, expected 6", i + 1, fields.len());
        }
        records.push(FamRecord {
            fid: fields[0].to_string(),
            iid: fields[1].to_string(),
            father: fields[2].to_string(),
            mother: fields[3].to_string(),
            sex: fields[4].parse()
                .with_context(|| format!("FAM line {}: invalid sex '{}'", i + 1, fields[4]))?,
            pheno: fields[5].to_string(),
        });
    }
    Ok(records)
}

fn parse_bim(path: &Path) -> Result<Vec<BimRecord>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 6 {
            bail!("BIM line {} has {} fields, expected 6", i + 1, fields.len());
        }
        records.push(BimRecord {
            chrom: fields[0].to_string(),
            snp_id: fields[1].to_string(),
            cm: fields[2].parse()
                .with_context(|| format!("BIM line {}: invalid cM '{}'", i + 1, fields[2]))?,
            pos: fields[3].parse()
                .with_context(|| format!("BIM line {}: invalid position '{}'", i + 1, fields[3]))?,
            allele1: fields[4].to_string(),
            allele2: fields[5].to_string(),
        });
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_roundtrip_encode_decode() {
        let n = 7; // odd number to test padding
        let p = 3;

        // Create genotype matrix
        let genotypes = Array2::from_shape_vec(
            (n, p),
            vec![
                0, 1, 2, // sample 0
                2, 0, 1, // sample 1
                1, 1, 0, // sample 2
                0, 2, 2, // sample 3
                2, 1, 0, // sample 4
                1, 0, 1, // sample 5
                0, 0, 2, // sample 6
            ],
        )
        .unwrap();

        let dir = TempDir::new().unwrap();
        let bed_path = dir.path().join("test.bed");

        // Write .bed
        write_bed_file(&bed_path, &genotypes).unwrap();

        // Read back raw bytes and decode
        let data = fs::read(&bed_path).unwrap();
        assert_eq!(data[0..3], BED_MAGIC);

        let packed = &data[3..];
        let decoded = decode_bed_chunk(packed, n, n, p, SnpNorm::Eigenstrat, None);

        // Verify decoded values are centered and Eigenstrat-scaled
        for snp in 0..p {
            let col: Vec<f64> = (0..n).map(|i| genotypes[(i, snp)] as f64).collect();
            let mean: f64 = col.iter().sum::<f64>() / n as f64;
            let af = mean / 2.0;
            let scale = 1.0 / (2.0 * af * (1.0 - af)).sqrt();
            for sample in 0..n {
                let expected = (col[sample] - mean) * scale;
                assert!(
                    (decoded[(sample, snp)] - expected).abs() < 1e-10,
                    "Mismatch at ({}, {}): got {}, expected {}",
                    sample,
                    snp,
                    decoded[(sample, snp)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_missing_imputation() {
        // Manually create .bed bytes with a missing value
        let n = 4;
        // SNP: sample0=0 (0b00), sample1=missing (0b01), sample2=1 (0b10), sample3=2 (0b11)
        // 4 samples = 1 byte per SNP
        let packed = vec![0b11_10_01_00u8];

        let decoded = decode_bed_chunk(&packed, n, n, 1, SnpNorm::Eigenstrat, None);

        // Mean of non-missing = (0 + 1 + 2) / 3 = 1.0
        // Allele freq p = 0.5, 2p(1-p) = 0.5, scale = 1/sqrt(0.5) = sqrt(2)
        // Centered: [-1, 0, 0, 1], scaled: [-sqrt(2), 0, 0, sqrt(2)]
        let s = std::f64::consts::SQRT_2;
        assert!((decoded[(0, 0)] - (-s)).abs() < 1e-10);
        assert!((decoded[(1, 0)] - 0.0).abs() < 1e-10); // imputed to mean, centered = 0
        assert!((decoded[(2, 0)] - 0.0).abs() < 1e-10);
        assert!((decoded[(3, 0)] - s).abs() < 1e-10);
    }

    #[test]
    fn test_normalization_modes() {
        // SNP: sample0=0, sample1=1, sample2=2 (no missing)
        // 3 samples = 1 byte per SNP
        // 0b00=0, 0b10=1, 0b11=2 → byte = 0b_00_11_10_00 = 0xE8... no wait:
        // sample0 bits [1:0]=00 (0), sample1 bits [3:2]=10 (1), sample2 bits [5:4]=11 (2)
        // byte = 0b__11_10_00 = 0x38
        let packed = vec![0b_11_10_00u8];
        let n = 3;

        // mean = (0+1+2)/3 = 1.0, p = 0.5, 2pq = 0.5
        let mean = 1.0_f64;
        let twopq = 0.5_f64;

        // CenterOnly: centered values, scale=1
        let dec = decode_bed_chunk(&packed, n, n, 1, SnpNorm::CenterOnly, None);
        assert!((dec[(0, 0)] - (0.0 - mean)).abs() < 1e-10);
        assert!((dec[(1, 0)] - (1.0 - mean)).abs() < 1e-10);
        assert!((dec[(2, 0)] - (2.0 - mean)).abs() < 1e-10);

        // Eigenstrat: (g - mean) / sqrt(2pq)
        let dec = decode_bed_chunk(&packed, n, n, 1, SnpNorm::Eigenstrat, None);
        let s = 1.0 / twopq.sqrt();
        assert!((dec[(0, 0)] - (0.0 - mean) * s).abs() < 1e-10);
        assert!((dec[(1, 0)] - (1.0 - mean) * s).abs() < 1e-10);
        assert!((dec[(2, 0)] - (2.0 - mean) * s).abs() < 1e-10);

        // HWE: (g - mean) * sqrt(2pq)
        let dec = decode_bed_chunk(&packed, n, n, 1, SnpNorm::Hwe, None);
        let s = twopq.sqrt();
        assert!((dec[(0, 0)] - (0.0 - mean) * s).abs() < 1e-10);
        assert!((dec[(1, 0)] - (1.0 - mean) * s).abs() < 1e-10);
        assert!((dec[(2, 0)] - (2.0 - mean) * s).abs() < 1e-10);
    }
}
