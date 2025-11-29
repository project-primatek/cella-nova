#!/usr/bin/env python3
"""
Download Protein-DNA Interaction Data
======================================

Downloads protein-DNA interaction data from multiple sources:
1. JASPAR - Transcription factor binding profiles
2. UniProt - DNA-binding proteins with annotations
3. ENCODE - ChIP-seq binding sites (optional)

Usage:
    python download_pdna.py --output data/pdna
    python download_pdna.py --output data/pdna --species human
"""

import argparse
import gzip
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

# Species taxonomy IDs
SPECIES_TAX_IDS = {
    "human": 9606,
    "mouse": 10090,
    "yeast": 559292,
    "fly": 7227,
    "worm": 6239,
    "zebrafish": 7955,
    "arabidopsis": 3702,
}

JASPAR_BASE_URL = "https://jaspar.elixir.no/api/v1"
UNIPROT_BASE_URL = "https://rest.uniprot.org"


def download_jaspar_motifs(output_dir: Path, tax_id: int = 9606) -> Path:
    """
    Download transcription factor binding motifs from JASPAR

    Args:
        output_dir: Output directory
        tax_id: NCBI taxonomy ID (default: human)

    Returns:
        Path to downloaded file
    """
    print("=" * 70)
    print("DOWNLOADING JASPAR TRANSCRIPTION FACTOR MOTIFS")
    print("=" * 70)
    print()

    jaspar_dir = output_dir / "jaspar"
    jaspar_dir.mkdir(parents=True, exist_ok=True)

    # Get list of all motifs for the species
    print(f"Fetching motifs for tax_id={tax_id}...")

    motifs = []
    page = 1
    page_size = 100

    while True:
        url = f"{JASPAR_BASE_URL}/matrix/"
        params = {
            "tax_id": tax_id,
            "page": page,
            "page_size": page_size,
            "format": "json",
            "collection": "CORE",  # CORE collection has well-curated motifs
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code != 200:
                print(f"Error fetching page {page}: {response.status_code}")
                break

            data = response.json()
            results = data.get("results", [])

            if not results:
                break

            motifs.extend(results)
            print(f"  Page {page}: {len(results)} motifs (total: {len(motifs)})")

            if not data.get("next"):
                break

            page += 1
            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            print(f"Error: {e}")
            break

    print(f"\n✓ Found {len(motifs)} motifs")

    # Download detailed information for each motif
    print("\nDownloading motif details and matrices...")

    motif_details = []
    pfms = {}  # Position Frequency Matrices

    for motif in tqdm(motifs, desc="Fetching details"):
        matrix_id = motif.get("matrix_id")
        if not matrix_id:
            continue

        try:
            # Get detailed motif info
            detail_url = f"{JASPAR_BASE_URL}/matrix/{matrix_id}/"
            response = requests.get(detail_url, timeout=10)

            if response.status_code == 200:
                detail = response.json()
                motif_details.append(detail)

                # Extract PFM (Position Frequency Matrix)
                pfm = detail.get("pfm")
                if pfm:
                    pfms[matrix_id] = pfm

            time.sleep(0.05)  # Rate limiting

        except Exception:
            continue

    # Save motif metadata
    metadata_file = jaspar_dir / "motifs_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(motif_details, f, indent=2)
    print(f"✓ Saved motif metadata: {metadata_file}")

    # Save PFMs in a more usable format
    pfm_file = jaspar_dir / "position_frequency_matrices.json"
    with open(pfm_file, "w") as f:
        json.dump(pfms, f, indent=2)
    print(f"✓ Saved PFMs: {pfm_file}")

    # Create a summary TSV file with TF info
    summary_file = jaspar_dir / "transcription_factors.tsv"
    with open(summary_file, "w") as f:
        f.write("matrix_id\tname\tuniprot_ids\tclass\tfamily\tspecies\n")
        for motif in motif_details:
            matrix_id = motif.get("matrix_id", "")
            name = motif.get("name", "")
            uniprot_ids = ",".join(motif.get("uniprot_ids", []))
            tf_class = motif.get("class", [""])[0] if motif.get("class") else ""
            family = motif.get("family", [""])[0] if motif.get("family") else ""
            species = (
                motif.get("species", [{}])[0].get("name", "")
                if motif.get("species")
                else ""
            )

            f.write(
                f"{matrix_id}\t{name}\t{uniprot_ids}\t{tf_class}\t{family}\t{species}\n"
            )

    print(f"✓ Saved TF summary: {summary_file}")

    # Generate binding site sequences from PFMs
    print("\nGenerating binding site sequences from PFMs...")
    sites_file = jaspar_dir / "binding_sites.tsv"

    def sample_sequences_from_pfm(pfm: dict, num_samples: int = 50) -> list:
        """Sample DNA sequences from a Position Frequency Matrix"""
        sequences = []
        nucleotides = ["A", "C", "G", "T"]

        # Get motif length
        length = len(pfm.get("A", []))
        if length == 0:
            return []

        # Convert PFM to probability matrix
        for _ in range(num_samples):
            seq = []
            for pos in range(length):
                counts = [pfm[nt][pos] for nt in nucleotides]
                total = sum(counts) + 1e-10
                probs = [c / total for c in counts]

                # Sample nucleotide based on probabilities
                nt = np.random.choice(nucleotides, p=probs)
                seq.append(nt)

            sequences.append("".join(seq))

        return sequences

    # Also generate consensus sequences
    def get_consensus(pfm: dict) -> str:
        """Get consensus sequence from PFM"""
        nucleotides = ["A", "C", "G", "T"]
        length = len(pfm.get("A", []))
        if length == 0:
            return ""

        consensus = []
        for pos in range(length):
            counts = [(pfm[nt][pos], nt) for nt in nucleotides]
            best_nt = max(counts, key=lambda x: x[0])[1]
            consensus.append(best_nt)

        return "".join(consensus)

    with open(sites_file, "w") as f:
        f.write("matrix_id\ttf_name\tsequence\n")

        for matrix_id, pfm in tqdm(pfms.items(), desc="Generating sequences"):
            # Find TF name for this matrix
            tf_name = ""
            for motif in motif_details:
                if motif.get("matrix_id") == matrix_id:
                    tf_name = motif.get("name", "")
                    break

            # Generate consensus
            consensus = get_consensus(pfm)
            if consensus:
                f.write(f"{matrix_id}\t{tf_name}\t{consensus}\n")

            # Sample additional sequences
            sampled = sample_sequences_from_pfm(pfm, num_samples=20)
            for seq in sampled:
                f.write(f"{matrix_id}\t{tf_name}\t{seq}\n")

    print(f"✓ Saved binding sites: {sites_file}")

    print(f"\n✓ JASPAR data saved to: {jaspar_dir}")
    return jaspar_dir


def download_uniprot_dna_binding_proteins(
    output_dir: Path, tax_id: int = 9606, max_proteins: int = None
) -> Path:
    """
    Download DNA-binding proteins from UniProt

    Args:
        output_dir: Output directory
        tax_id: NCBI taxonomy ID
        max_proteins: Maximum number of proteins to download

    Returns:
        Path to output directory
    """
    print()
    print("=" * 70)
    print("DOWNLOADING UNIPROT DNA-BINDING PROTEINS")
    print("=" * 70)
    print()

    uniprot_dir = output_dir / "uniprot"
    uniprot_dir.mkdir(parents=True, exist_ok=True)

    # Search for DNA-binding proteins
    # Using GO term for DNA binding (GO:0003677) and taxonomy filter
    query = f"(go:0003677) AND (taxonomy_id:{tax_id}) AND (reviewed:true)"

    print(f"Searching UniProt for DNA-binding proteins...")
    print(f"  Query: {query}")

    # First, get the count
    search_url = f"{UNIPROT_BASE_URL}/uniprotkb/search"
    params = {
        "query": query,
        "format": "json",
        "size": 1,
    }

    response = requests.get(search_url, params=params, timeout=30)
    if response.status_code != 200:
        print(f"Error searching UniProt: {response.status_code}")
        return None

    # Get total count from headers
    total_count = int(response.headers.get("x-total-results", 0))
    print(f"  Found {total_count:,} DNA-binding proteins")

    if max_proteins:
        total_count = min(total_count, max_proteins)
        print(f"  Limiting to {total_count:,} proteins")

    # Download proteins in batches
    print("\nDownloading protein data...")

    proteins = []
    batch_size = 500
    cursor = None

    with tqdm(total=total_count, desc="Downloading") as pbar:
        while len(proteins) < total_count:
            params = {
                "query": query,
                "format": "json",
                "size": min(batch_size, total_count - len(proteins)),
                "fields": "accession,id,gene_names,protein_name,sequence,ft_dna_bind,go,organism_name,length",
            }

            if cursor:
                params["cursor"] = cursor

            try:
                response = requests.get(search_url, params=params, timeout=60)

                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    break

                data = response.json()
                results = data.get("results", [])

                if not results:
                    break

                proteins.extend(results)
                pbar.update(len(results))

                # Get next cursor from Link header
                link_header = response.headers.get("link", "")
                if 'rel="next"' in link_header:
                    # Extract cursor from link
                    import re

                    match = re.search(r"cursor=([^&>]+)", link_header)
                    if match:
                        cursor = match.group(1)
                    else:
                        break
                else:
                    break

                time.sleep(0.1)

            except Exception as e:
                print(f"Error: {e}")
                break

    print(f"\n✓ Downloaded {len(proteins):,} proteins")

    # Save protein data
    proteins_file = uniprot_dir / "dna_binding_proteins.json"
    with open(proteins_file, "w") as f:
        json.dump(proteins, f, indent=2)
    print(f"✓ Saved protein data: {proteins_file}")

    # Save sequences in FASTA format
    fasta_file = uniprot_dir / "dna_binding_proteins.fasta"
    with open(fasta_file, "w") as f:
        for protein in proteins:
            accession = protein.get("primaryAccession", "")
            gene_name = ""
            if protein.get("genes"):
                gene_name = protein["genes"][0].get("geneName", {}).get("value", "")

            sequence = protein.get("sequence", {}).get("value", "")

            if accession and sequence:
                f.write(f">{accession}|{gene_name}\n")
                # Write sequence in lines of 60 characters
                for i in range(0, len(sequence), 60):
                    f.write(sequence[i : i + 60] + "\n")

    print(f"✓ Saved sequences: {fasta_file}")

    # Extract DNA-binding domains
    print("\nExtracting DNA-binding domain annotations...")

    domains_file = uniprot_dir / "dna_binding_domains.tsv"
    with open(domains_file, "w") as f:
        f.write("accession\tgene_name\tdomain_type\tstart\tend\tdescription\n")

        for protein in proteins:
            accession = protein.get("primaryAccession", "")
            gene_name = ""
            if protein.get("genes"):
                gene_name = protein["genes"][0].get("geneName", {}).get("value", "")

            # Get DNA binding features
            features = protein.get("features", [])
            for feature in features:
                if feature.get("type") == "DNA binding":
                    start = (
                        feature.get("location", {}).get("start", {}).get("value", "")
                    )
                    end = feature.get("location", {}).get("end", {}).get("value", "")
                    description = feature.get("description", "")

                    f.write(
                        f"{accession}\t{gene_name}\tDNA_binding\t{start}\t{end}\t{description}\n"
                    )

    print(f"✓ Saved domain annotations: {domains_file}")

    # Create summary file
    summary_file = uniprot_dir / "summary.tsv"
    with open(summary_file, "w") as f:
        f.write(
            "accession\tgene_name\tprotein_name\tlength\torganism\thas_dna_binding_domain\n"
        )

        for protein in proteins:
            accession = protein.get("primaryAccession", "")
            gene_name = ""
            if protein.get("genes"):
                gene_name = protein["genes"][0].get("geneName", {}).get("value", "")

            protein_name = (
                protein.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", "")
            )
            if not protein_name:
                protein_name = (
                    protein.get("proteinDescription", {})
                    .get("submissionNames", [{}])[0]
                    .get("fullName", {})
                    .get("value", "")
                )

            length = protein.get("sequence", {}).get("length", "")
            organism = protein.get("organism", {}).get("scientificName", "")

            has_dna_binding = (
                "Yes"
                if any(
                    f.get("type") == "DNA binding" for f in protein.get("features", [])
                )
                else "No"
            )

            f.write(
                f"{accession}\t{gene_name}\t{protein_name}\t{length}\t{organism}\t{has_dna_binding}\n"
            )

    print(f"✓ Saved summary: {summary_file}")

    print(f"\n✓ UniProt data saved to: {uniprot_dir}")
    return uniprot_dir


def download_encode_chip_seq(output_dir: Path, max_experiments: int = 100) -> Path:
    """
    Download ChIP-seq binding data from ENCODE

    Args:
        output_dir: Output directory
        max_experiments: Maximum number of experiments to download

    Returns:
        Path to output directory
    """
    print()
    print("=" * 70)
    print("DOWNLOADING ENCODE CHIP-SEQ DATA")
    print("=" * 70)
    print()

    encode_dir = output_dir / "encode"
    encode_dir.mkdir(parents=True, exist_ok=True)

    # Search for human TF ChIP-seq experiments
    encode_url = "https://www.encodeproject.org/search/"
    params = {
        "type": "Experiment",
        "assay_title": "TF ChIP-seq",
        "replicates.library.biosample.donor.organism.scientific_name": "Homo sapiens",
        "status": "released",
        "format": "json",
        "limit": max_experiments,
        "frame": "object",
    }

    print("Searching ENCODE for TF ChIP-seq experiments...")

    try:
        response = requests.get(
            encode_url,
            params=params,
            timeout=60,
            headers={"Accept": "application/json"},
        )

        if response.status_code != 200:
            print(f"Error searching ENCODE: {response.status_code}")
            return None

        data = response.json()
        experiments = data.get("@graph", [])

        print(f"  Found {len(experiments)} experiments")

    except Exception as e:
        print(f"Error: {e}")
        return None

    # Save experiment metadata
    experiments_file = encode_dir / "experiments.json"
    with open(experiments_file, "w") as f:
        json.dump(experiments, f, indent=2)
    print(f"✓ Saved experiment metadata: {experiments_file}")

    # Extract TF and target information
    targets_file = encode_dir / "tf_targets.tsv"
    with open(targets_file, "w") as f:
        f.write("experiment_id\ttarget_gene\ttarget_label\tcell_type\tbiosample\n")

        for exp in experiments:
            exp_id = exp.get("accession", "")
            target = exp.get("target", {})

            if isinstance(target, dict):
                target_gene = target.get("gene_name", "")
                target_label = target.get("label", "")
            else:
                target_gene = ""
                target_label = ""

            # Get biosample info
            biosample = ""
            cell_type = ""
            replicates = exp.get("replicates", [])
            if replicates:
                lib = replicates[0].get("library", {})
                if lib:
                    bs = lib.get("biosample", {})
                    if bs:
                        biosample = bs.get("summary", "")
                        cell_type = bs.get("biosample_ontology", {}).get(
                            "term_name", ""
                        )

            f.write(
                f"{exp_id}\t{target_gene}\t{target_label}\t{cell_type}\t{biosample}\n"
            )

    print(f"✓ Saved TF targets: {targets_file}")

    # Download peak files for a subset of experiments
    print("\nNote: Full peak file download requires additional setup.")
    print("Peak files can be downloaded from ENCODE portal for specific experiments.")

    print(f"\n✓ ENCODE data saved to: {encode_dir}")
    return encode_dir


def create_training_data(output_dir: Path) -> Path:
    """
    Create training data by combining JASPAR and UniProt data

    Args:
        output_dir: Output directory containing downloaded data

    Returns:
        Path to training data file
    """
    print()
    print("=" * 70)
    print("CREATING TRAINING DATA")
    print("=" * 70)
    print()

    jaspar_dir = output_dir / "jaspar"
    uniprot_dir = output_dir / "uniprot"

    if not jaspar_dir.exists() or not uniprot_dir.exists():
        print("Error: JASPAR and UniProt data must be downloaded first")
        return None

    # Load JASPAR TF data with UniProt IDs
    tf_file = jaspar_dir / "transcription_factors.tsv"
    jaspar_tfs = {}
    tf_name_to_matrix = {}  # Map TF name to matrix IDs

    with open(tf_file, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                matrix_id = parts[0]
                name = parts[1].upper()  # Normalize to uppercase
                uniprot_ids = parts[2].split(",") if len(parts) > 2 and parts[2] else []

                jaspar_tfs[matrix_id] = {
                    "name": name,
                    "uniprot_ids": [u.strip() for u in uniprot_ids if u.strip()],
                }

                # Build reverse mapping
                if name not in tf_name_to_matrix:
                    tf_name_to_matrix[name] = []
                tf_name_to_matrix[name].append(matrix_id)

    print(f"Loaded {len(jaspar_tfs)} JASPAR transcription factors")

    # Load binding sites
    sites_file = jaspar_dir / "binding_sites.tsv"
    binding_sites = []
    matrix_to_sites = {}

    with open(sites_file, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                matrix_id = parts[0]
                tf_name = parts[1]
                sequence = parts[2]

                binding_sites.append(
                    {
                        "matrix_id": matrix_id,
                        "tf_name": tf_name,
                        "sequence": sequence,
                    }
                )

                if matrix_id not in matrix_to_sites:
                    matrix_to_sites[matrix_id] = []
                matrix_to_sites[matrix_id].append(sequence)

    print(f"Loaded {len(binding_sites)} binding site sequences")

    # Load UniProt sequences and build gene name mapping
    fasta_file = uniprot_dir / "dna_binding_proteins.fasta"
    uniprot_sequences = {}
    gene_to_uniprot = {}  # Map gene name to UniProt ID

    current_id = None
    current_gene = None
    current_seq = []

    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    uniprot_sequences[current_id] = "".join(current_seq)
                    if current_gene:
                        gene_to_uniprot[current_gene.upper()] = current_id

                # Parse header: >P12345|GENE_NAME
                header = line[1:]
                parts = header.split("|")
                current_id = parts[0]
                current_gene = parts[1] if len(parts) > 1 else None
                current_seq = []
            else:
                current_seq.append(line)

        if current_id:
            uniprot_sequences[current_id] = "".join(current_seq)
            if current_gene:
                gene_to_uniprot[current_gene.upper()] = current_id

    print(f"Loaded {len(uniprot_sequences)} UniProt sequences")
    print(f"Built gene name mappings for {len(gene_to_uniprot)} genes")

    # Create positive pairs using multiple mapping strategies
    training_file = output_dir / "protein_dna_interactions.tsv"
    positive_pairs = set()  # Use set to avoid duplicates

    print("\nCreating protein-DNA pairs...")

    with open(training_file, "w") as f:
        f.write("protein_id\tprotein_sequence\tdna_sequence\tlabel\tsource\n")

        for site in tqdm(binding_sites, desc="Processing"):
            matrix_id = site["matrix_id"]
            tf_name = site["tf_name"].upper()
            dna_seq = site["sequence"]

            if len(dna_seq) < 6:
                continue

            found_protein = False

            if matrix_id in jaspar_tfs:
                tf_info = jaspar_tfs[matrix_id]

                # Strategy 1: Use direct UniProt IDs from JASPAR
                for uniprot_id in tf_info["uniprot_ids"]:
                    if uniprot_id in uniprot_sequences:
                        protein_seq = uniprot_sequences[uniprot_id]
                        if len(protein_seq) >= 50:
                            pair_key = (uniprot_id, dna_seq)
                            if pair_key not in positive_pairs:
                                f.write(
                                    f"{uniprot_id}\t{protein_seq}\t{dna_seq}\t1\tJASPAR_direct\n"
                                )
                                positive_pairs.add(pair_key)
                                found_protein = True

                # Strategy 2: Map TF name to UniProt via gene name
                if not found_protein and tf_name in gene_to_uniprot:
                    uniprot_id = gene_to_uniprot[tf_name]
                    protein_seq = uniprot_sequences[uniprot_id]
                    if len(protein_seq) >= 50:
                        pair_key = (uniprot_id, dna_seq)
                        if pair_key not in positive_pairs:
                            f.write(
                                f"{uniprot_id}\t{protein_seq}\t{dna_seq}\t1\tJASPAR_gene\n"
                            )
                            positive_pairs.add(pair_key)
                            found_protein = True

                # Strategy 3: Try partial gene name match (e.g., "STAT1" in "STAT1::STAT2")
                if not found_protein:
                    for gene_name, uniprot_id in gene_to_uniprot.items():
                        if gene_name in tf_name or tf_name in gene_name:
                            protein_seq = uniprot_sequences[uniprot_id]
                            if len(protein_seq) >= 50:
                                pair_key = (uniprot_id, dna_seq)
                                if pair_key not in positive_pairs:
                                    f.write(
                                        f"{uniprot_id}\t{protein_seq}\t{dna_seq}\t1\tJASPAR_partial\n"
                                    )
                                    positive_pairs.add(pair_key)
                                    found_protein = True
                                    break

    print(f"\n✓ Created {len(positive_pairs)} positive protein-DNA pairs")
    print(f"✓ Saved training data: {training_file}")

    # Save summary
    summary = {
        "num_transcription_factors": len(jaspar_tfs),
        "num_binding_sites": len(binding_sites),
        "num_uniprot_proteins": len(uniprot_sequences),
        "num_positive_pairs": len(positive_pairs),
        "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_file = output_dir / "data_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved summary: {summary_file}")

    return training_file


def main():
    parser = argparse.ArgumentParser(
        description="Download protein-DNA interaction data"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/pdna",
        help="Output directory (default: data/pdna)",
    )

    parser.add_argument(
        "--species",
        type=str,
        default="human",
        choices=list(SPECIES_TAX_IDS.keys()),
        help="Species to download data for (default: human)",
    )

    parser.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        help="Maximum number of proteins to download from UniProt",
    )

    parser.add_argument(
        "--skip-jaspar",
        action="store_true",
        help="Skip JASPAR download",
    )

    parser.add_argument(
        "--skip-uniprot",
        action="store_true",
        help="Skip UniProt download",
    )

    parser.add_argument(
        "--skip-encode",
        action="store_true",
        help="Skip ENCODE download",
    )

    parser.add_argument(
        "--include-encode",
        action="store_true",
        help="Include ENCODE ChIP-seq data (large download)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tax_id = SPECIES_TAX_IDS.get(args.species, 9606)

    print("=" * 70)
    print("PROTEIN-DNA INTERACTION DATA DOWNLOAD")
    print("=" * 70)
    print()
    print(f"Species: {args.species} (tax_id: {tax_id})")
    print(f"Output directory: {output_dir}")
    print()

    # Download JASPAR data
    if not args.skip_jaspar:
        download_jaspar_motifs(output_dir, tax_id=tax_id)

    # Download UniProt data
    if not args.skip_uniprot:
        download_uniprot_dna_binding_proteins(
            output_dir, tax_id=tax_id, max_proteins=args.max_proteins
        )

    # Download ENCODE data (optional, large)
    if args.include_encode and not args.skip_encode:
        download_encode_chip_seq(output_dir)

    # Create training data
    create_training_data(output_dir)

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print()
    print(f"✓ All data saved to: {output_dir}")
    print()
    print("Files:")
    print("  • jaspar/transcription_factors.tsv - TF information")
    print("  • jaspar/binding_sites.tsv - DNA binding sequences")
    print("  • jaspar/position_frequency_matrices.json - Motif PWMs")
    print("  • uniprot/dna_binding_proteins.fasta - Protein sequences")
    print("  • uniprot/dna_binding_domains.tsv - Domain annotations")
    print("  • protein_dna_interactions.tsv - Training data")
    print()


if __name__ == "__main__":
    main()
