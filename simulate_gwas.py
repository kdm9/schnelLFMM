#!/usr/bin/env python3
"""Simulate realistic GWAS data: genotypes (plink .bed/.bim/.fam) and phenotypes (.tsv).

Genotypes are simulated via msprime (coalescent with recombination) from K ancestral
populations with configurable Fst. Phenotypes are constructed from causal variants
with Laplace-distributed effect sizes scaled to target heritability.

Dependencies: msprime, numpy, scipy
"""

import argparse
import csv
import math
import struct
import sys
from pathlib import Path

import msprime
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Simulate GWAS genotypes and phenotypes")
    p.add_argument("--out", required=True, help="Output basename (prefix for all files)")
    p.add_argument("--n-samples", type=int, default=1000, help="Total number of diploid samples")
    p.add_argument("--n-snps", type=int, default=10000, help="Target number of SNPs")
    p.add_argument("--k-pops", type=int, default=3, help="Number of ancestral populations")
    p.add_argument(
        "--pop-props",
        type=str,
        default=None,
        help="Comma-separated population proportions (must sum to 1). Default: equal.",
    )
    p.add_argument("--fst", type=float, default=0.1, help="Target pairwise Fst between populations")
    p.add_argument(
        "--traits",
        required=True,
        help="CSV file with columns: name, heritability, n_causal",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def parse_pop_props(pop_props_str, k_pops):
    if pop_props_str is None:
        return [1.0 / k_pops] * k_pops
    props = [float(x) for x in pop_props_str.split(",")]
    if len(props) != k_pops:
        sys.exit(f"Error: --pop-props has {len(props)} values but --k-pops is {k_pops}")
    s = sum(props)
    if abs(s - 1.0) > 0.01:
        sys.exit(f"Error: --pop-props sum to {s}, expected ~1.0")
    # Normalize to exactly 1
    props = [p / s for p in props]
    return props


def parse_traits(path):
    traits = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            traits.append(
                {
                    "name": row["name"].strip(),
                    "heritability": float(row["heritability"]),
                    "n_causal": int(row["n_causal"]),
                }
            )
    return traits


def build_demography(k_pops, fst, ne=10000):
    """Build a demographic model with K populations splitting from an ancestor.

    Divergence times are calibrated so that pairwise Fst ~ target value.
    Fst ≈ 1 - exp(-T / (2*Ne)) for the simple island model divergence,
    so T ≈ -2*Ne * ln(1 - Fst).
    """
    demography = msprime.Demography()
    demography.add_population(name="ANC", initial_size=ne)
    for i in range(k_pops):
        demography.add_population(name=f"POP{i}", initial_size=ne)

    # Split times: stagger slightly so populations aren't all identical
    # Base time from Fst, then space them out
    if fst >= 1.0:
        t_base = 20 * ne
    else:
        t_base = max(1, -2 * ne * math.log(1 - fst))

    for i in range(k_pops):
        # Stagger: most recent split at t_base, oldest at t_base * 1.5
        frac = (i / max(1, k_pops - 1)) * 0.5 + 1.0  # range [1.0, 1.5]
        t_split = t_base * frac
        demography.add_population_split(
            time=t_split, derived=[f"POP{i}"], ancestral="ANC"
        )

    return demography


def simulate_genotypes(n_samples, n_snps_target, k_pops, pop_props, fst, seed):
    """Simulate genotypes with msprime and return dosage matrix + variant info."""
    rng = np.random.default_rng(seed)
    demography = build_demography(k_pops, fst)

    # Compute per-population sample counts
    counts = []
    remaining = n_samples
    for i in range(k_pops - 1):
        c = max(1, round(n_samples * pop_props[i]))
        counts.append(c)
        remaining -= c
    counts.append(max(1, remaining))
    total = sum(counts)

    samples = []
    sample_ids = []
    idx = 0
    for i, c in enumerate(counts):
        samples.append(msprime.SampleSet(c, population=f"POP{i}", ploidy=2))
        for j in range(c):
            sample_ids.append(f"POP{i}_{idx}")
            idx += 1

    pop_labels = []
    for i, c in enumerate(counts):
        pop_labels.extend([i] * c)

    # Choose sequence length to get roughly the right number of SNPs.
    # With mutation_rate=1e-8, recomb_rate=1e-8, and Ne=10000,
    # expected #segregating sites ~ 4*Ne*mu*L * harmonic(2n-1) for n samples.
    # Rough estimate: ~4e-4 * L for 1000 samples. Target L = n_snps / 4e-4.
    # We'll overshoot by 1.5x and then downsample.
    harmonic_n = sum(1.0 / i for i in range(1, 2 * total))
    theta_per_bp = 4 * 10000 * 1e-8  # 4*Ne*mu
    expected_snps_per_bp = theta_per_bp * harmonic_n
    seq_length = int(n_snps_target * 1.5 / max(expected_snps_per_bp, 1e-10))
    seq_length = max(seq_length, 100_000)  # minimum 100kb

    print(f"Simulating ancestry: {total} samples, seq_length={seq_length:,}bp ...")
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        sequence_length=seq_length,
        recombination_rate=1e-8,
        random_seed=rng.integers(1, 2**31),
    )

    print("Overlaying mutations ...")
    ts = msprime.sim_mutations(
        ts, rate=1e-8, random_seed=rng.integers(1, 2**31)
    )

    n_variants = ts.num_sites
    print(f"Got {n_variants} variant sites")

    # Extract genotype dosage matrix (n_samples x n_variants)
    print("Extracting genotypes ...")
    positions = []
    alleles_list = []
    geno_rows = []  # will be list of arrays, one per SNP

    for var in ts.variants():
        # Only biallelic SNPs
        if len(var.alleles) != 2:
            continue
        if any(len(a) != 1 for a in var.alleles):
            continue  # skip indels
        # Compute dosage per individual (sum of two haplotypes)
        haps = var.genotypes  # length 2*n_samples
        dosage = haps[0::2] + haps[1::2]  # 0, 1, or 2
        # MAF filter
        af = dosage.sum() / (2 * len(dosage))
        maf = min(af, 1 - af)
        if maf < 0.01:
            continue
        geno_rows.append(dosage.astype(np.uint8))
        positions.append(int(var.site.position))
        alleles_list.append(var.alleles)

    n_passing = len(geno_rows)
    print(f"{n_passing} biallelic SNPs pass MAF filter")

    if n_passing == 0:
        sys.exit("Error: no SNPs passed filters. Try increasing --n-snps or sequence length.")

    # Downsample if needed
    if n_passing > n_snps_target:
        idx = np.sort(rng.choice(n_passing, n_snps_target, replace=False))
        geno_rows = [geno_rows[i] for i in idx]
        positions = [positions[i] for i in idx]
        alleles_list = [alleles_list[i] for i in idx]
        print(f"Thinned to {n_snps_target} SNPs")
    elif n_passing < n_snps_target:
        print(f"Warning: only {n_passing} SNPs available (target was {n_snps_target})")

    G = np.array(geno_rows).T  # n_samples x n_snps
    return G, positions, alleles_list, sample_ids, pop_labels


def simulate_phenotypes(G, traits, rng):
    """Simulate phenotypes from genotype matrix G.

    Returns (phenotypes, causal_info) where:
      phenotypes: dict of {trait_name: y_vector}
      causal_info: list of (trait_name, snp_index, effect_size) tuples
    """
    n_samples, n_snps = G.shape
    phenotypes = {}
    causal_info = []

    for trait in traits:
        name = trait["name"]
        h2 = trait["heritability"]
        n_causal = min(trait["n_causal"], n_snps)

        # Pick causal SNPs
        causal_idx = rng.choice(n_snps, n_causal, replace=False)
        G_causal = G[:, causal_idx].astype(np.float64)

        # Standardize causal genotypes (mean-center, unit variance)
        means = G_causal.mean(axis=0)
        stds = G_causal.std(axis=0)
        stds[stds == 0] = 1.0
        G_std = (G_causal - means) / stds

        # Draw effect sizes from Laplace distribution
        beta = rng.laplace(loc=0, scale=1.0, size=n_causal)

        # Compute genetic values
        g = G_std @ beta

        # Scale to target heritability
        var_g = np.var(g)
        if var_g == 0 or h2 == 0:
            # No genetic signal
            y = rng.normal(0, 1, n_samples)
        else:
            # Var(e) = Var(g) * (1 - h2) / h2
            var_e = var_g * (1 - h2) / h2
            noise = rng.normal(0, np.sqrt(var_e), n_samples)
            y = g + noise

        phenotypes[name] = y
        actual_h2 = var_g / np.var(y) if np.var(y) > 0 else 0
        print(f"  Trait '{name}': {n_causal} causal SNPs, target h2={h2:.3f}, actual h2={actual_h2:.3f}")

        for ci, b in zip(causal_idx, beta):
            causal_info.append((name, int(ci), float(b)))

    return phenotypes, causal_info


def write_fam(path, sample_ids):
    with open(path, "w") as f:
        for sid in sample_ids:
            # FID IID father mother sex phenotype
            f.write(f"{sid}\t{sid}\t0\t0\t0\t-9\n")


def write_bim(path, positions, alleles_list):
    with open(path, "w") as f:
        for i, (pos, alleles) in enumerate(zip(positions, alleles_list)):
            # chr  snp_id  genetic_dist  bp_pos  allele1  allele2
            f.write(f"1\tsnp_{i}\t0\t{pos}\t{alleles[0]}\t{alleles[1]}\n")


def write_bed(path, G):
    """Write plink .bed file in SNP-major mode.

    G is n_samples x n_snps, values 0/1/2.
    Plink encoding (2 bits per sample, LSB first within each byte):
      00 = homozygous A1/A1 (dosage 0)
      10 = heterozygous       (dosage 1)
      11 = homozygous A2/A2  (dosage 2)
      01 = missing
    """
    n_samples, n_snps = G.shape
    bytes_per_snp = math.ceil(n_samples / 4)

    # Encoding lookup: dosage -> 2-bit code
    encode = np.array([0b00, 0b10, 0b11, 0b01], dtype=np.uint8)  # 0->00, 1->10, 2->11, missing->01

    with open(path, "wb") as f:
        # Magic bytes: SNP-major mode
        f.write(struct.pack("BBB", 0x6C, 0x1B, 0x01))

        for snp_j in range(n_snps):
            col = G[:, snp_j]  # n_samples values, each 0/1/2
            # Map dosage to 2-bit codes
            codes = encode[col]  # safe because values are 0,1,2

            # Pack 4 samples per byte, LSB first
            # Pad to multiple of 4
            padded = np.zeros(bytes_per_snp * 4, dtype=np.uint8)
            padded[:n_samples] = codes
            # Reshape and pack
            groups = padded.reshape(-1, 4)
            packed = (
                groups[:, 0]
                | (groups[:, 1] << 2)
                | (groups[:, 2] << 4)
                | (groups[:, 3] << 6)
            )
            f.write(packed.tobytes())


def write_causal(path, causal_info, positions, alleles_list):
    """Write causal variant ground truth to a TSV file."""
    with open(path, "w") as f:
        f.write("trait\tsnp_index\tsnp_id\tchr\tposition\tallele1\tallele2\teffect_size\n")
        for trait_name, snp_idx, beta in causal_info:
            pos = positions[snp_idx]
            a1, a2 = alleles_list[snp_idx]
            f.write(f"{trait_name}\t{snp_idx}\tsnp_{snp_idx}\t1\t{pos}\t{a1}\t{a2}\t{beta:.6f}\n")


def write_phenotypes(path, sample_ids, phenotypes):
    trait_names = list(phenotypes.keys())
    with open(path, "w") as f:
        f.write("sample_id\t" + "\t".join(trait_names) + "\n")
        for i, sid in enumerate(sample_ids):
            vals = "\t".join(f"{phenotypes[t][i]:.6f}" for t in trait_names)
            f.write(f"{sid}\t{vals}\n")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    pop_props = parse_pop_props(args.pop_props, args.k_pops)
    traits = parse_traits(args.traits)

    print(f"Populations: {args.k_pops}, proportions: {[f'{p:.2f}' for p in pop_props]}")
    print(f"Target Fst: {args.fst}")
    print(f"Traits: {[t['name'] for t in traits]}")

    G, positions, alleles_list, sample_ids, pop_labels = simulate_genotypes(
        n_samples=args.n_samples,
        n_snps_target=args.n_snps,
        k_pops=args.k_pops,
        pop_props=pop_props,
        fst=args.fst,
        seed=rng.integers(1, 2**31),
    )

    print(f"\nFinal genotype matrix: {G.shape[0]} samples x {G.shape[1]} SNPs")

    print("\nSimulating phenotypes ...")
    phenotypes, causal_info = simulate_phenotypes(G, traits, rng)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting output files with prefix '{args.out}' ...")
    write_fam(f"{args.out}.fam", sample_ids)
    write_bim(f"{args.out}.bim", positions, alleles_list)
    write_bed(f"{args.out}.bed", G)
    write_phenotypes(f"{args.out}_phenotypes.tsv", sample_ids, phenotypes)
    write_causal(f"{args.out}_causal.tsv", causal_info, positions, alleles_list)

    print("Done.")
    print(f"  {args.out}.bed / .bim / .fam")
    print(f"  {args.out}_phenotypes.tsv")
    print(f"  {args.out}_causal.tsv")


if __name__ == "__main__":
    main()
