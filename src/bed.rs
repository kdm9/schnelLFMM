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

        // Validate magic bytes
        if mmap.len() < 3 || mmap[0..3] != BED_MAGIC {
            bail!(
                "Invalid .bed file (bad magic bytes): {}",
                bed_path.display()
            );
        }

        // Validate file size
        let bytes_per_snp = (n_samples + 3) / 4;
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
            n_samples,
            n_snps,
            bim_records,
            fam_records,
            mmap,
        })
    }

    /// Bytes per SNP in the .bed file (ceil(n_samples / 4))
    pub fn bytes_per_snp(&self) -> usize {
        (self.n_samples + 3) / 4
    }

    /// Total number of SNPs that will be yielded by this subset
    pub fn subset_snp_count(&self, subset: &SubsetSpec) -> usize {
        match subset {
            SubsetSpec::All => self.n_snps,
            SubsetSpec::Rate(rate) => {
                let step = (1.0 / rate).ceil() as usize;
                (self.n_snps + step - 1) / step
            }
            SubsetSpec::Indices(indices) => indices.len(),
        }
    }
}

/// Decode a chunk of SNPs from .bed format into a centered f64 matrix.
/// Missing values (0b01) are imputed to the per-SNP mean genotype, then centered.
pub fn decode_bed_chunk(packed: &[u8], n_samples: usize, chunk_size: usize) -> Array2<f64> {
    let bytes_per_snp = (n_samples + 3) / 4;
    let mut out = Array2::<f64>::zeros((n_samples, chunk_size));

    for snp in 0..chunk_size {
        let snp_bytes = &packed[snp * bytes_per_snp..(snp + 1) * bytes_per_snp];
        decode_single_snp(snp_bytes, n_samples, out.column_mut(snp));
    }
    out
}

/// Decode specific SNPs from mmap data into a pre-allocated output matrix.
///
/// - `mmap_data`: mmap bytes after the 3-byte header
/// - `bps`: bytes per SNP
/// - `n_samples`: number of samples
/// - `snp_indices`: which SNPs to decode (column indices in the .bed file)
/// - `out`: pre-allocated output slice (n_samples × snp_indices.len())
pub fn decode_bed_chunk_into(
    mmap_data: &[u8],
    bps: usize,
    n_samples: usize,
    snp_indices: &[usize],
    mut out: ndarray::ArrayViewMut2<f64>,
) {
    for (col, &snp_idx) in snp_indices.iter().enumerate() {
        let snp_bytes = &mmap_data[snp_idx * bps..(snp_idx + 1) * bps];
        decode_single_snp(snp_bytes, n_samples, out.column_mut(col));
    }
}

/// Decode a single SNP column, impute missing to mean, and center.
pub(crate) fn decode_single_snp(
    snp_bytes: &[u8],
    n_samples: usize,
    mut col: ndarray::ArrayViewMut1<f64>,
) {
    // Pass 1: decode and compute mean (excluding missing)
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

    let mean = if n_valid > 0 {
        sum / n_valid as f64
    } else {
        0.0
    };

    // Pass 2: impute missing -> mean, then center all values
    for sample in 0..n_samples {
        if col[sample].is_nan() {
            col[sample] = 0.0; // missing -> mean, centered = 0
        } else {
            col[sample] -= mean;
        }
    }
}

/// Write a PLINK .bed file from genotype matrix (n_samples × n_snps, values in {0,1,2}).
pub fn write_bed_file(path: &Path, genotypes: &Array2<u8>) -> Result<()> {
    let n_samples = genotypes.nrows();
    let n_snps = genotypes.ncols();
    let bytes_per_snp = (n_samples + 3) / 4;

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
            sex: fields[4].parse().unwrap_or(0),
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
            cm: fields[2].parse().unwrap_or(0.0),
            pos: fields[3].parse().unwrap_or(0),
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
        let decoded = decode_bed_chunk(packed, n, p);

        // Verify decoded values are centered versions of original
        for snp in 0..p {
            let col: Vec<f64> = (0..n).map(|i| genotypes[(i, snp)] as f64).collect();
            let mean: f64 = col.iter().sum::<f64>() / n as f64;
            for sample in 0..n {
                let expected = col[sample] - mean;
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

        let decoded = decode_bed_chunk(&packed, n, 1);

        // Mean of non-missing = (0 + 1 + 2) / 3 = 1.0
        // After centering: 0-1=-1, missing->0, 1-1=0, 2-1=1
        assert!((decoded[(0, 0)] - (-1.0)).abs() < 1e-10);
        assert!((decoded[(1, 0)] - 0.0).abs() < 1e-10); // imputed to mean, centered = 0
        assert!((decoded[(2, 0)] - 0.0).abs() < 1e-10);
        assert!((decoded[(3, 0)] - 1.0).abs() < 1e-10);
    }
}
