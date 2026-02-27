#!/usr/bin/env python3
"""Simulate realistic GWAS data: genotypes (plink .bed/.bim/.fam) and phenotypes (.tsv).

Three simulation methods:
  msprime        - coalescent with recombination (realistic but slow for large datasets)
  mvnorm         - block-correlated multivariate normal with Balding-Nichols population
                   structure. Streams block-by-block so datasets larger than RAM can be
                   generated. LD block sizes vary along the chromosome by drawing rho
                   from a Gamma-based distribution.
  real_genotypes - use an existing PLINK .bed file and simulate phenotypes from
                   causal SNPs drawn from the real genotypes. Does not write new
                   PLINK files.

Phenotypes are constructed from causal variants with Laplace-distributed effect
sizes scaled to target heritability.

Dependencies: numpy, scipy (+ msprime if using --sim-method msprime)
"""

import argparse
import csv
import math
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
try:
    from tqdm.auto import tqdm

except ImportError:
    def tqdm(x, *args, **kwargs):
        yield from x



# ---------------------------------------------------------------------------
# CLI and parsing
# ---------------------------------------------------------------------------

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
    p.add_argument(
        "--sim-method",
        choices=["msprime", "mvnorm", "real_genotypes"],
        default="mvnorm",
        help="Genotype simulation method (default: mvnorm). "
        "real_genotypes requires --bed pointing to an existing PLINK .bed file.",
    )
    p.add_argument(
        "--bed",
        type=str,
        default=None,
        help="Path to existing PLINK .bed file (required for --sim-method real_genotypes). "
        "Matching .bim and .fam files must exist alongside it.",
    )
    p.add_argument(
        "--ld-decay",
        type=float,
        default=0.95,
        help="Mean LD decay parameter rho for mvnorm method. "
        "Correlation between adjacent SNPs = rho, decays as rho^|i-j|. "
        "Block sizes are computed so r2 falls below 10%% at block boundaries. "
        "Rho varies per block via a Gamma distribution centered on this value. "
        "(default: 0.95)",
    )
    p.add_argument(
        "--max-block-size",
        type=int,
        default=100000,
        help="Maximum SNPs per LD block for mvnorm method (default: 100000)",
    )
    p.add_argument(
        "--causal-regions",
        type=str,
        default=None,
        help="BED or GFF file specifying genomic regions where causal SNPs may be placed "
        "(e.g. chromosome arms excluding centromeres). Only used with --sim-method real_genotypes. "
        "BED format: chr, start (0-based), end. GFF format auto-detected by column count.",
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


def parse_causal_regions(path):
    """Parse a BED or GFF file into a dict of {chr: [(start, end), ...]}.

    BED: 0-based half-open (chr, start, end, ...).
    GFF/GFF3/GTF: 1-based inclusive (chr, source, type, start, end, ...),
    converted to 0-based half-open internally.
    Format is auto-detected: lines with >=8 tab-separated fields where
    columns 4 and 5 are integers are treated as GFF; otherwise BED.
    """
    regions = {}  # chr -> list of (start, end)  0-based half-open
    is_gff = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            fields = line.split("\t")

            # Auto-detect format on first data line
            if is_gff is None:
                if len(fields) >= 8:
                    try:
                        int(fields[3])
                        int(fields[4])
                        is_gff = True
                    except ValueError:
                        is_gff = False
                else:
                    is_gff = False

            if is_gff:
                chrom = fields[0]
                start = int(fields[3]) - 1  # GFF is 1-based inclusive -> 0-based
                end = int(fields[4])         # end stays (1-based inclusive end == 0-based half-open end)
            else:
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])

            regions.setdefault(chrom, []).append((start, end))

    # Sort intervals per chromosome
    for chrom in regions:
        regions[chrom].sort()

    total_regions = sum(len(v) for v in regions.values())
    total_bp = sum(e - s for intervals in regions.values() for s, e in intervals)
    print(f"Loaded {total_regions} causal regions from {path} "
          f"across {len(regions)} chromosomes ({total_bp:,} bp total)")
    return regions


def snp_in_regions(chrom, pos, regions):
    """Check if a SNP (chrom, 1-based pos) falls within any region.

    Uses binary search for efficiency. Regions are 0-based half-open.
    """
    import bisect
    intervals = regions.get(str(chrom))
    if intervals is None:
        return False
    # pos is 1-based from BIM; convert to 0-based for comparison
    pos0 = pos - 1
    # Find the rightmost interval whose start <= pos0
    idx = bisect.bisect_right(intervals, (pos0, float('inf'))) - 1
    if idx < 0:
        return False
    start, end = intervals[idx]
    return start <= pos0 < end


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def compute_pop_counts(n_samples, k_pops, pop_props):
    counts = []
    remaining = n_samples
    for i in range(k_pops - 1):
        c = max(1, round(n_samples * pop_props[i]))
        counts.append(c)
        remaining -= c
    counts.append(max(1, remaining))
    return counts


def make_sample_ids(counts):
    sample_ids = []
    idx = 0
    for i, c in enumerate(counts):
        for _ in range(c):
            sample_ids.append(f"POP{i}_{idx}")
            idx += 1
    return sample_ids


BED_ENCODE = np.array([0b00, 0b10, 0b11, 0b01], dtype=np.uint8)


def encode_bed_snps(G_block, n_samples):
    """Encode a genotype block (n_samples x n_snps_block) into .bed bytes.

    Returns bytes for all SNPs in the block (SNP-major order).
    """
    n_snps = G_block.shape[1]
    bytes_per_snp = math.ceil(n_samples / 4)
    pad_to = bytes_per_snp * 4

    buf = bytearray(bytes_per_snp * n_snps)
    for j in range(n_snps):
        codes = BED_ENCODE[G_block[:, j]]
        padded = np.zeros(pad_to, dtype=np.uint8)
        padded[:n_samples] = codes
        groups = padded.reshape(-1, 4)
        packed = (
            groups[:, 0]
            | (groups[:, 1] << 2)
            | (groups[:, 2] << 4)
            | (groups[:, 3] << 6)
        )
        offset = j * bytes_per_snp
        buf[offset : offset + bytes_per_snp] = packed.tobytes()
    return bytes(buf)


# ---------------------------------------------------------------------------
# mvnorm streaming simulation
# ---------------------------------------------------------------------------

def compute_block_size_from_rho(rho, r2_threshold=0.1):
    """Number of SNPs until r2 falls below threshold for AR(1) with param rho.

    r2(d) = rho^(2d).  Solve rho^(2d) = threshold => d = log(threshold) / (2*log(rho)).
    """
    if rho <= 0 or rho >= 1:
        return 1
    return max(1, math.ceil(math.log(r2_threshold) / (2 * math.log(rho))))


def plan_block_schedule(n_snps, ld_decay, max_block_size, rng):
    """Plan LD blocks with varying rho.

    Recombination rate r is drawn from Gamma(shape=2, scale=r_mean/2) so the mean
    is r_mean = -log(ld_decay). This gives natural variation: most blocks near the
    target LD, with occasional recombination hotspots (high r -> small blocks).
    rho_block = exp(-r), block_size = ceil(log(0.1) / (2*log(rho))).
    """
    r_mean = -math.log(max(ld_decay, 1e-10))
    schedule = []  # list of (block_size, rho_block)
    total = 0
    while total < n_snps:
        r = rng.gamma(shape=2, scale=r_mean / 2)
        r = max(r, 1e-6)
        rho_block = math.exp(-r)
        bs = compute_block_size_from_rho(rho_block)
        bs = min(bs, max_block_size, n_snps - total)
        bs = max(bs, 1)
        schedule.append((bs, rho_block))
        total += bs
    return schedule


def simulate_one_block(bs, rho, counts, p_pop_block, rng):
    """Simulate genotypes for one LD block across all populations.

    Returns G_block of shape (total_samples, bs) as uint8 dosage (0/1/2).
    """
    total = sum(counts)
    k_pops = len(counts)

    # AR(1) Cholesky
    idx = np.arange(bs)
    R = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(R)

    G_block = np.zeros((total, bs), dtype=np.uint8)
    sample_offset = 0
    for pop_k in range(k_pops):
        n_k = counts[pop_k]
        freqs = p_pop_block[pop_k]  # (bs,)
        for _hap in range(2):
            z = rng.standard_normal((n_k, bs))
            corr_z = z @ L.T
            u = norm.cdf(corr_z)
            G_block[sample_offset : sample_offset + n_k] += (u < freqs[None, :]).astype(np.uint8)
        sample_offset += n_k

    return G_block


def simulate_and_write_mvnorm(out, n_samples, n_snps, k_pops, pop_props, fst,
                               ld_decay, max_block_size, traits, rng):
    """Streaming mvnorm simulation: plan blocks, simulate each, write, free.

    Memory usage is O(n_samples * max_block_size) per block, not O(n_samples * n_snps).
    Phenotype causal contributions are accumulated incrementally.
    """
    counts = compute_pop_counts(n_samples, k_pops, pop_props)
    total = sum(counts)
    sample_ids = make_sample_ids(counts)

    # --- Phase 1: Plan block schedule ---
    schedule = plan_block_schedule(n_snps, ld_decay, max_block_size, rng)
    total_planned = sum(bs for bs, _ in schedule)
    n_blocks = len(schedule)
    block_sizes = [bs for bs, _ in schedule]
    block_rhos = [rho for _, rho in schedule]

    print(f"Planned {n_blocks} LD blocks, {total_planned} SNPs total")
    print(f"  block sizes: min={min(block_sizes)}, median={sorted(block_sizes)[n_blocks//2]}, "
          f"max={max(block_sizes)}")
    print(f"  rho range: [{min(block_rhos):.4f}, {max(block_rhos):.4f}]")

    # --- Phase 2: Pre-select causal SNPs (in pre-filter index space) ---
    # For each trait, choose which of the total_planned SNPs are causal and
    # draw their effect sizes. If a causal SNP turns out monomorphic, it's skipped.
    causal_sets = {}  # trait_name -> {pre_filter_idx: beta}
    for trait in traits:
        n_causal = min(trait["n_causal"], total_planned)
        indices = rng.choice(total_planned, n_causal, replace=False)
        betas = rng.laplace(0, 1.0, size=n_causal)
        causal_sets[trait["name"]] = dict(zip(indices.tolist(), betas.tolist()))

    # --- Phase 3: Stream blocks ---
    write_fam(f"{out}.fam", sample_ids)

    fst_safe = np.clip(fst, 0.001, 0.999)
    fst_scale = (1 - fst_safe) / fst_safe

    genetic_values = {t["name"]: np.zeros(total) for t in traits}
    causal_info = []  # (trait_name, post_filter_idx, beta) for output file
    positions_all = []
    alleles_all = []

    n_written = 0
    pre_filter_offset = 0
    bases = ["A", "C", "G", "T"]

    with open(f"{out}.bed", "wb") as f_bed, open(f"{out}.bim", "w") as f_bim:
        f_bed.write(struct.pack("BBB", 0x6C, 0x1B, 0x01))

        for block_idx, (bs, rho_block) in tqdm(enumerate(schedule), unit="block", total=len(schedule)):
            # Generate allele frequencies for this block
            p_anc_block = rng.beta(0.5, 0.5, size=bs)
            p_anc_block = np.clip(p_anc_block, 0.01, 0.99)
            p_pop_block = np.zeros((k_pops, bs))
            for k in range(k_pops):
                a = p_anc_block * fst_scale
                b = (1 - p_anc_block) * fst_scale
                p_pop_block[k] = rng.beta(a, b)
            p_pop_block = np.clip(p_pop_block, 0.001, 0.999)

            # Simulate genotypes for this block
            G_block = simulate_one_block(bs, rho_block, counts, p_pop_block, rng)

            # Filter monomorphic SNPs
            af = G_block.sum(axis=0) / (2 * total)
            maf = np.minimum(af, 1 - af)
            poly_mask = maf > 0

            # Pre-filter indices for this block
            pre_indices = np.arange(pre_filter_offset, pre_filter_offset + bs)
            surviving_pre = pre_indices[poly_mask]
            G_surviving = G_block[:, poly_mask]
            n_surviving = G_surviving.shape[1]

            if n_surviving > 0:
                # Write .bed for surviving SNPs
                f_bed.write(encode_bed_snps(G_surviving, total))

                # Write .bim and build position/allele lists
                for j in range(n_surviving):
                    post_idx = n_written + j
                    pos = post_idx * 500 + 1
                    a1, a2 = bases[post_idx % 4], bases[(post_idx + 1) % 4]
                    f_bim.write(f"1\tsnp_{post_idx}\t0\t{pos}\t{a1}\t{a2}\n")
                    positions_all.append(pos)
                    alleles_all.append((a1, a2))

                # Accumulate genetic values for any causal SNPs in this block
                for trait in traits:
                    cs = causal_sets[trait["name"]]
                    for j, pre_idx in enumerate(surviving_pre):
                        pre_idx_int = int(pre_idx)
                        if pre_idx_int in cs:
                            beta = cs[pre_idx_int]
                            col = G_surviving[:, j].astype(np.float64)
                            std = col.std()
                            if std > 0:
                                col = (col - col.mean()) / std
                            else:
                                col = col - col.mean()
                            genetic_values[trait["name"]] += col * beta
                            causal_info.append((trait["name"], n_written + j, beta))

                n_written += n_surviving

            pre_filter_offset += bs

            #if (block_idx + 1) % max(1, n_blocks // 10) == 0 or block_idx == n_blocks - 1:
            #    print(f"  Block {block_idx + 1}/{n_blocks}: "
            #          f"{n_written} SNPs written so far")

    print(f"\n{n_written} / {total_planned} polymorphic SNPs written")

    # --- Phase 4: Compute phenotypes ---
    print("\nSimulating phenotypes ...")
    phenotypes = {}
    for trait in traits:
        name = trait["name"]
        h2 = trait["heritability"]
        g = genetic_values[name]
        var_g = np.var(g)

        if var_g == 0 or h2 == 0:
            y = rng.normal(0, 1, total)
        else:
            var_e = var_g * (1 - h2) / h2
            noise = rng.normal(0, np.sqrt(var_e), total)
            y = g + noise

        phenotypes[name] = y
        actual_h2 = var_g / np.var(y) if np.var(y) > 0 else 0
        n_causal_actual = sum(1 for t, _, _ in causal_info if t == name)
        print(f"  Trait '{name}': {n_causal_actual} causal SNPs (of {trait['n_causal']} requested), "
              f"target h2={h2:.3f}, actual h2={actual_h2:.3f}")

    write_phenotypes(f"{out}_phenotypes.tsv", sample_ids, phenotypes)
    write_causal_streaming(f"{out}_causal.tsv", causal_info, positions_all, alleles_all)

    print(f"\nDone.")
    print(f"  {out}.bed / .bim / .fam")
    print(f"  {out}_phenotypes.tsv")
    print(f"  {out}_causal.tsv")


# ---------------------------------------------------------------------------
# msprime simulation (in-memory, unchanged)
# ---------------------------------------------------------------------------

def simulate_genotypes_msprime(n_samples, n_snps_target, k_pops, pop_props, fst, seed):
    """Simulate genotypes with msprime and return dosage matrix + variant info."""
    import msprime

    rng = np.random.default_rng(seed)
    ne = 10000

    demography = msprime.Demography()
    demography.add_population(name="ANC", initial_size=ne)
    for i in range(k_pops):
        demography.add_population(name=f"POP{i}", initial_size=ne)

    if fst >= 1.0:
        t_base = 20 * ne
    else:
        t_base = max(1, -2 * ne * math.log(1 - fst))

    for i in range(k_pops):
        frac = (i / max(1, k_pops - 1)) * 0.5 + 1.0
        t_split = t_base * frac
        demography.add_population_split(
            time=t_split, derived=[f"POP{i}"], ancestral="ANC"
        )

    counts = compute_pop_counts(n_samples, k_pops, pop_props)
    total = sum(counts)
    sample_ids = make_sample_ids(counts)

    samples = [
        msprime.SampleSet(c, population=f"POP{i}", ploidy=2)
        for i, c in enumerate(counts)
    ]

    harmonic_n = sum(1.0 / i for i in range(1, 2 * total))
    theta_per_bp = 4 * ne * 1e-8
    expected_snps_per_bp = theta_per_bp * harmonic_n
    seq_length = int(n_snps_target * 1.5 / max(expected_snps_per_bp, 1e-10))
    seq_length = max(seq_length, 100_000)

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

    print(f"Got {ts.num_sites} variant sites")
    print("Extracting genotypes ...")

    positions = []
    alleles_list = []
    geno_rows = []

    for var in ts.variants():
        if len(var.alleles) != 2:
            continue
        if any(len(a) != 1 for a in var.alleles):
            continue
        haps = var.genotypes
        dosage = haps[0::2] + haps[1::2]
        af = dosage.sum() / (2 * len(dosage))
        maf = min(af, 1 - af)
        if maf == 0:
            continue
        geno_rows.append(dosage.astype(np.uint8))
        positions.append(int(var.site.position))
        alleles_list.append(var.alleles)

    n_passing = len(geno_rows)
    print(f"{n_passing} polymorphic biallelic SNPs retained")

    if n_passing == 0:
        sys.exit("Error: no SNPs passed filters. Try increasing --n-snps or sequence length.")

    if n_passing > n_snps_target:
        idx = np.sort(rng.choice(n_passing, n_snps_target, replace=False))
        geno_rows = [geno_rows[i] for i in idx]
        positions = [positions[i] for i in idx]
        alleles_list = [alleles_list[i] for i in idx]
        print(f"Thinned to {n_snps_target} SNPs")
    elif n_passing < n_snps_target:
        print(f"Warning: only {n_passing} SNPs available (target was {n_snps_target})")

    G = np.array(geno_rows).T
    return G, positions, alleles_list, sample_ids


# ---------------------------------------------------------------------------
# real_genotypes simulation
# ---------------------------------------------------------------------------

def read_bed_genotypes(bed_path):
    """Read a PLINK .bed file and return (G, bim_records, sample_ids).

    G is (n_samples x n_snps) uint8 dosage matrix (0/1/2, 255=missing).
    bim_records is a list of (chr, snp_id, pos, a1, a2) per SNP.
    """
    bed_path = Path(bed_path)
    fam_path = bed_path.with_suffix(".fam")
    bim_path = bed_path.with_suffix(".bim")

    if not bed_path.exists():
        sys.exit(f"Error: BED file not found: {bed_path}")
    if not fam_path.exists():
        sys.exit(f"Error: FAM file not found: {fam_path}")
    if not bim_path.exists():
        sys.exit(f"Error: BIM file not found: {bim_path}")

    # Read FAM -> sample IDs (IID = column 1)
    sample_ids = []
    with open(fam_path) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) >= 2:
                sample_ids.append(fields[1])  # IID
    n_samples = len(sample_ids)

    # Read BIM -> (chr, snp_id, pos, a1, a2)
    bim_records = []
    with open(bim_path) as f:
        for line in f:
            fields = line.strip().split()
            bim_records.append((fields[0], fields[1], int(fields[3]), fields[4], fields[5]))
    n_snps = len(bim_records)

    # Read BED
    BED_DECODE = np.array([0, 255, 1, 2], dtype=np.uint8)  # 00->hom_ref, 01->missing, 10->het, 11->hom_alt
    bytes_per_snp = math.ceil(n_samples / 4)

    with open(bed_path, "rb") as f:
        magic = f.read(3)
        if magic != b"\x6c\x1b\x01":
            sys.exit("Error: not a valid SNP-major PLINK .bed file")

        G = np.zeros((n_samples, n_snps), dtype=np.uint8)
        for j in range(n_snps):
            raw = np.frombuffer(f.read(bytes_per_snp), dtype=np.uint8)
            # Unpack 2-bit genotypes
            g0 = raw & 0x03
            g1 = (raw >> 2) & 0x03
            g2 = (raw >> 4) & 0x03
            g3 = (raw >> 6) & 0x03
            unpacked = np.empty(bytes_per_snp * 4, dtype=np.uint8)
            unpacked[0::4] = g0
            unpacked[1::4] = g1
            unpacked[2::4] = g2
            unpacked[3::4] = g3
            G[:, j] = BED_DECODE[unpacked[:n_samples]]

    return G, bim_records, sample_ids


def simulate_real_genotypes(bed_path, traits, out, rng, causal_regions=None):
    """Use real genotypes from an existing PLINK file, simulate phenotypes."""
    print(f"Reading genotypes from {bed_path} ...")
    G, bim_records, sample_ids = read_bed_genotypes(bed_path)
    n_samples, n_snps = G.shape
    print(f"  {n_samples} samples, {n_snps} SNPs")

    # Replace missing (255) with per-SNP mean for phenotype simulation
    G_float = G.astype(np.float64)
    for j in range(n_snps):
        col = G_float[:, j]
        missing = col == 255
        if missing.any():
            mean_val = col[~missing].mean() if (~missing).any() else 0
            col[missing] = mean_val

    # Filter to polymorphic SNPs for causal selection
    af = G_float.mean(axis=0) / 2
    maf = np.minimum(af, 1 - af)
    poly_mask = maf > 0.15  # require MAF > 15% for causal SNPs
    poly_indices = np.where(poly_mask)[0]
    print(f"  {len(poly_indices)} SNPs with MAF > 15% available as causal candidates")

    # Further restrict to allowed causal regions if specified
    if causal_regions is not None:
        region_mask = np.array([
            snp_in_regions(bim_records[i][0], bim_records[i][2], causal_regions)
            for i in poly_indices
        ])
        poly_indices = poly_indices[region_mask]
        print(f"  {len(poly_indices)} of those fall within causal regions")

    print("\nSimulating phenotypes ...")
    phenotypes = {}
    causal_info = []

    for trait in traits:
        name = trait["name"]
        h2 = trait["heritability"]
        n_causal = min(trait["n_causal"], len(poly_indices))

        causal_poly_idx = rng.choice(len(poly_indices), n_causal, replace=False)
        causal_idx = poly_indices[causal_poly_idx]

        G_causal = G_float[:, causal_idx]
        means = G_causal.mean(axis=0)
        stds = G_causal.std(axis=0)
        stds[stds == 0] = 1.0
        G_std = (G_causal - means) / stds

        # Truncated Laplace: resample any effects with |beta| < min_effect
        min_effect = 0.05
        beta = rng.laplace(loc=0, scale=1.0, size=n_causal)
        small = np.abs(beta) < min_effect
        while small.any():
            beta[small] = rng.laplace(loc=0, scale=1.0, size=small.sum())
            small = np.abs(beta) < min_effect
        g = G_std @ beta

        var_g = np.var(g)
        if var_g == 0 or h2 == 0:
            y = rng.normal(0, 1, n_samples)
        else:
            var_e = var_g * (1 - h2) / h2
            noise = rng.normal(0, np.sqrt(var_e), n_samples)
            y = g + noise

        phenotypes[name] = y
        actual_h2 = var_g / np.var(y) if np.var(y) > 0 else 0
        print(f"  Trait '{name}': {n_causal} causal SNPs, target h2={h2:.3f}, actual h2={actual_h2:.3f}")

        for ci, b in zip(causal_idx, beta):
            causal_info.append((name, int(ci), float(b)))

    write_phenotypes(f"{out}_phenotypes.tsv", sample_ids, phenotypes)

    # Write causal file using real chr/snp_id/pos/alleles from BIM
    with open(f"{out}_causal.tsv", "w") as f:
        f.write("trait\tsnp_index\tsnp_id\tchr\tposition\tallele1\tallele2\teffect_size\n")
        for trait_name, snp_idx, beta in causal_info:
            chrom, snp_id, pos, a1, a2 = bim_records[snp_idx]
            f.write(f"{trait_name}\t{snp_idx}\t{snp_id}\t{chrom}\t{pos}\t{a1}\t{a2}\t{beta:.6f}\n")

    print(f"\nDone.")
    print(f"  {out}_phenotypes.tsv")
    print(f"  {out}_causal.tsv")


# ---------------------------------------------------------------------------
# Phenotype simulation (in-memory, for msprime path)
# ---------------------------------------------------------------------------

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

        causal_idx = rng.choice(n_snps, n_causal, replace=False)
        G_causal = G[:, causal_idx].astype(np.float64)

        means = G_causal.mean(axis=0)
        stds = G_causal.std(axis=0)
        stds[stds == 0] = 1.0
        G_std = (G_causal - means) / stds

        beta = rng.laplace(loc=0, scale=1.0, size=n_causal)
        g = G_std @ beta

        var_g = np.var(g)
        if var_g == 0 or h2 == 0:
            y = rng.normal(0, 1, n_samples)
        else:
            var_e = var_g * (1 - h2) / h2
            noise = rng.normal(0, np.sqrt(var_e), n_samples)
            y = g + noise

        phenotypes[name] = y
        actual_h2 = var_g / np.var(y) if np.var(y) > 0 else 0
        print(f"  Trait '{name}': {n_causal} causal SNPs, target h2={h2:.3f}, actual h2={actual_h2:.3f}")

        for ci, b in zip(causal_idx, beta):
            causal_info.append((name, int(ci), float(b)))

    return phenotypes, causal_info


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_fam(path, sample_ids):
    with open(path, "w") as f:
        for sid in sample_ids:
            f.write(f"{sid}\t{sid}\t0\t0\t0\t-9\n")


def write_bim(path, positions, alleles_list):
    with open(path, "w") as f:
        for i, (pos, alleles) in enumerate(zip(positions, alleles_list)):
            f.write(f"1\tsnp_{i}\t0\t{pos}\t{alleles[0]}\t{alleles[1]}\n")


def write_bed(path, G):
    """Write plink .bed file in SNP-major mode."""
    n_samples = G.shape[0]
    with open(path, "wb") as f:
        f.write(struct.pack("BBB", 0x6C, 0x1B, 0x01))
        f.write(encode_bed_snps(G, n_samples))


def write_causal(path, causal_info, positions, alleles_list):
    """Write causal variant ground truth to a TSV file."""
    with open(path, "w") as f:
        f.write("trait\tsnp_index\tsnp_id\tchr\tposition\tallele1\tallele2\teffect_size\n")
        for trait_name, snp_idx, beta in causal_info:
            pos = positions[snp_idx]
            a1, a2 = alleles_list[snp_idx]
            f.write(f"{trait_name}\t{snp_idx}\tsnp_{snp_idx}\t1\t{pos}\t{a1}\t{a2}\t{beta:.6f}\n")


def write_causal_streaming(path, causal_info, positions, alleles_list):
    """Write causal variant ground truth (streaming version uses accumulated lists)."""
    with open(path, "w") as f:
        f.write("trait\tsnp_index\tsnp_id\tchr\tposition\tallele1\tallele2\teffect_size\n")
        for trait_name, post_idx, beta in causal_info:
            pos = positions[post_idx]
            a1, a2 = alleles_list[post_idx]
            f.write(f"{trait_name}\t{post_idx}\tsnp_{post_idx}\t1\t{pos}\t{a1}\t{a2}\t{beta:.6f}\n")


def write_phenotypes(path, sample_ids, phenotypes):
    trait_names = list(phenotypes.keys())
    with open(path, "w") as f:
        f.write("sample_id\t" + "\t".join(trait_names) + "\n")
        for i, sid in enumerate(sample_ids):
            vals = "\t".join(f"{phenotypes[t][i]:.6f}" for t in trait_names)
            f.write(f"{sid}\t{vals}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    pop_props = parse_pop_props(args.pop_props, args.k_pops)
    traits = parse_traits(args.traits)

    print(f"Method: {args.sim_method}")
    print(f"Traits: {[t['name'] for t in traits]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.sim_method == "real_genotypes":
        if args.bed is None:
            sys.exit("Error: --bed is required for --sim-method real_genotypes")
        causal_regions = None
        if args.causal_regions is not None:
            causal_regions = parse_causal_regions(args.causal_regions)
        simulate_real_genotypes(
            bed_path=args.bed,
            traits=traits,
            out=args.out,
            rng=rng,
            causal_regions=causal_regions,
        )
        return

    print(f"Populations: {args.k_pops}, proportions: {[f'{p:.2f}' for p in pop_props]}")
    print(f"Target Fst: {args.fst}")

    if args.sim_method == "mvnorm":
        simulate_and_write_mvnorm(
            out=args.out,
            n_samples=args.n_samples,
            n_snps=args.n_snps,
            k_pops=args.k_pops,
            pop_props=pop_props,
            fst=args.fst,
            ld_decay=args.ld_decay,
            max_block_size=args.max_block_size,
            traits=traits,
            rng=rng,
        )
    else:
        G, positions, alleles_list, sample_ids = simulate_genotypes_msprime(
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
