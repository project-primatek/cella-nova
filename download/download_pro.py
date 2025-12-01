#!/usr/bin/env python3
"""
Download Protein Data
=====================

Unified script for downloading protein data from multiple sources:
1. UniProt - Protein sequences and annotations
2. AlphaFold - Predicted 3D structures
3. STRING - Protein network information

This script provides the protein data foundation for all interaction models.

Usage:
    python download_pro.py --species "Homo sapiens"
    python download_pro.py --taxon-id 9606 --include-structures
    python download_pro.py --species "Homo sapiens" --filter dna-binding
    python download_pro.py --species "Homo sapiens" --filter rna-binding
    python download_pro.py --uniprot-ids P53_HUMAN,BRCA1_HUMAN

Filters:
    - all: Download all proteins in proteome (default)
    - dna-binding: Only DNA-binding proteins (GO:0003677)
    - rna-binding: Only RNA-binding proteins (KW-0694)
    - membrane: Only membrane proteins (KW-0472)
    - enzyme: Only enzymes (KW-0378)
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# API ENDPOINTS
# =============================================================================

# UniProt
UNIPROT_BASE_URL = "https://rest.uniprot.org"
UNIPROT_SEARCH_URL = f"{UNIPROT_BASE_URL}/uniprotkb/search"
UNIPROT_STREAM_URL = f"{UNIPROT_BASE_URL}/uniprotkb/stream"
UNIPROT_PROTEOMES_URL = f"{UNIPROT_BASE_URL}/proteomes/search"

# AlphaFold
ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files"

# STRING
STRING_API_URL = "https://string-db.org/api/json"
STRING_DOWNLOAD_URL = "https://stringdb-downloads.org/download"
STRING_VERSION = "12.0"

# STRING uses species-level taxon IDs, not strain-specific ones
# Map strain taxon IDs to STRING-compatible species taxon IDs
STRING_TAXON_ID_MAP = {
    559292: 4932,  # S. cerevisiae S288C -> S. cerevisiae (species)
    83333: 511145,  # E. coli K12 -> E. coli K-12 MG1655 (STRING)
    # Add more mappings as needed
}

# =============================================================================
# FILTER DEFINITIONS
# =============================================================================

PROTEIN_FILTERS = {
    "all": None,  # No filter, get all proteins
    "dna-binding": "(go:0003677)",  # DNA binding GO term
    "rna-binding": "(keyword:KW-0694)",  # RNA-binding keyword
    "membrane": "(keyword:KW-0472)",  # Membrane keyword
    "enzyme": "(keyword:KW-0378)",  # Enzyme keyword
    "kinase": "(keyword:KW-0418)",  # Kinase keyword
    "receptor": "(keyword:KW-0675)",  # Receptor keyword
    "transcription-factor": "(go:0003700)",  # TF activity GO term
    "drug-target": "(annotation:(type:pharmaceutical))",  # Pharmaceutical annotation
}


# =============================================================================
# UNIPROT FUNCTIONS
# =============================================================================


def search_proteome(species_name: str, strain: Optional[str] = None) -> Optional[Dict]:
    """
    Search for proteome by species name

    Args:
        species_name: Species name (e.g., "Homo sapiens")
        strain: Optional strain name

    Returns:
        Proteome information dictionary or None if not found
    """
    print(f"Searching for proteome: {species_name}")
    if strain:
        print(f"  Strain: {strain}")
    print()

    search_query = species_name
    if strain:
        search_query += f" {strain}"

    params = {"query": search_query, "format": "json", "size": 10}

    try:
        response = requests.get(UNIPROT_PROTEOMES_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Error searching proteomes: {e}")
        return None

    results = response.json()

    if not results.get("results"):
        print(f"❌ No proteomes found for: {species_name}")
        return None

    print(f"Found {len(results['results'])} proteome(s):")
    print()

    for i, proteome in enumerate(results["results"], 1):
        upid = proteome.get("id", "Unknown")
        organism = proteome.get("taxonomy", {}).get("scientificName", "Unknown")
        protein_count = proteome.get("proteinCount", 0)
        is_reference = proteome.get("isReferenceProteome", False)
        taxon_id = proteome.get("taxonomy", {}).get("taxonId", "Unknown")

        ref_marker = " [REFERENCE]" if is_reference else ""

        print(f"{i}. {upid} - {organism}{ref_marker}")
        print(f"   Taxon ID: {taxon_id}")
        print(f"   Proteins: {protein_count:,}")
        print()

    # Return first (usually best match, reference proteome)
    return results["results"][0]


def get_taxon_id_from_species(species_name: str) -> Optional[int]:
    """Get NCBI taxon ID from species name using UniProt"""
    proteome = search_proteome(species_name)
    if proteome:
        return proteome.get("taxonomy", {}).get("taxonId")
    return None


def build_uniprot_query(
    taxon_id: int,
    filter_type: str = "all",
    reviewed_only: bool = True,
    proteome_id: Optional[str] = None,
) -> str:
    """
    Build UniProt search query

    Args:
        taxon_id: NCBI taxonomy ID
        filter_type: Type of protein filter
        reviewed_only: Only include reviewed (Swiss-Prot) entries
        proteome_id: Optional proteome ID for more specific query

    Returns:
        UniProt query string
    """
    query_parts = []

    # Taxon filter
    query_parts.append(f"(taxonomy_id:{taxon_id})")

    # Proteome filter (more specific)
    if proteome_id:
        query_parts.append(f"(proteome:{proteome_id})")

    # Reviewed filter
    if reviewed_only:
        query_parts.append("(reviewed:true)")

    # Protein type filter
    if filter_type != "all" and filter_type in PROTEIN_FILTERS:
        filter_query = PROTEIN_FILTERS[filter_type]
        if filter_query:
            query_parts.append(filter_query)

    return " AND ".join(query_parts)


def download_proteins(
    output_dir: Path,
    taxon_id: int,
    filter_type: str = "all",
    reviewed_only: bool = True,
    proteome_id: Optional[str] = None,
    max_proteins: Optional[int] = None,
    include_features: bool = True,
) -> Optional[Dict[str, Dict]]:
    """
    Download protein data from UniProt

    Args:
        output_dir: Output directory
        taxon_id: NCBI taxonomy ID
        filter_type: Type of protein filter
        reviewed_only: Only include reviewed entries
        proteome_id: Optional proteome ID
        max_proteins: Maximum number of proteins to download
        include_features: Include domain/feature annotations

    Returns:
        Dictionary mapping UniProt IDs to protein data
    """
    print("=" * 70)
    print("DOWNLOADING PROTEINS FROM UNIPROT")
    print("=" * 70)
    print()

    query = build_uniprot_query(taxon_id, filter_type, reviewed_only, proteome_id)
    print(f"Query: {query}")
    print()

    # Fields to retrieve
    fields = [
        "accession",
        "id",
        "gene_names",
        "protein_name",
        "sequence",
        "length",
        "organism_name",
        "organism_id",
    ]

    if include_features:
        fields.extend(
            [
                "ft_domain",
                "ft_binding",
                "ft_act_site",
                "go_p",
                "go_f",
                "go_c",
                "keyword",
            ]
        )

    # First, get count
    count_params = {"query": query, "format": "json", "size": 1}
    total_results = 0

    for attempt in range(3):
        try:
            response = requests.get(UNIPROT_SEARCH_URL, params=count_params, timeout=30)
            response.raise_for_status()

            # Get total from headers
            total_results = int(response.headers.get("x-total-results", 0))
            if total_results > 0:
                break
        except requests.RequestException as e:
            if attempt == 2:
                print(f"❌ Error querying UniProt: {e}")
                return None
            time.sleep(1)

    if total_results == 0:
        print("❌ No proteins found matching query")
        return None

    print(f"Found {total_results:,} proteins matching query")

    if max_proteins:
        total_results = min(total_results, max_proteins)
        print(f"Limiting to {total_results:,} proteins")
    print()

    # Download proteins using cursor-based pagination (UniProt REST API)
    proteins: Dict[str, Dict] = {}
    batch_size = 500
    target_count = total_results

    print(f"Starting download of {target_count:,} proteins...")

    pbar = tqdm(
        total=target_count,
        desc="Downloading",
        unit=" proteins",
        dynamic_ncols=True,
    )

    # Initial request URL
    next_url = f"{UNIPROT_SEARCH_URL}?query={requests.utils.quote(query)}&format=json&fields={','.join(fields)}&size={batch_size}"

    while next_url:
        try:
            response = requests.get(next_url, timeout=60)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                break

            for entry in results:
                accession = entry.get("primaryAccession", "")
                if not accession:
                    continue

                # Extract sequence
                sequence = entry.get("sequence", {}).get("value", "")
                if not sequence:
                    continue

                # Extract gene name
                gene_name = ""
                genes = entry.get("genes", [])
                if genes:
                    gene_name = genes[0].get("geneName", {}).get("value", "")

                # Extract protein name
                protein_name = ""
                prot_desc = entry.get("proteinDescription", {})
                rec_name = prot_desc.get("recommendedName", {})
                if rec_name:
                    protein_name = rec_name.get("fullName", {}).get("value", "")

                # Build protein entry
                proteins[accession] = {
                    "accession": accession,
                    "entry_name": entry.get("uniProtkbId", ""),
                    "gene_name": gene_name,
                    "protein_name": protein_name,
                    "sequence": sequence,
                    "length": len(sequence),
                    "organism": entry.get("organism", {}).get("scientificName", ""),
                    "taxon_id": entry.get("organism", {}).get("taxonId", taxon_id),
                }

                # Add features if requested
                if include_features:
                    proteins[accession]["keywords"] = [
                        kw.get("name", "") for kw in entry.get("keywords", [])
                    ]
                    proteins[accession]["go_terms"] = {
                        "process": entry.get("goTerms", {}).get("P", []),
                        "function": entry.get("goTerms", {}).get("F", []),
                        "component": entry.get("goTerms", {}).get("C", []),
                    }

                if max_proteins and len(proteins) >= max_proteins:
                    break

            pbar.update(len(results))

            # Check if we've hit max_proteins limit
            if max_proteins and len(proteins) >= max_proteins:
                break

            # Get next page URL from Link header (cursor-based pagination)
            next_url = None
            link_header = response.headers.get("Link", "")
            if 'rel="next"' in link_header:
                # Parse: <URL>; rel="next"
                # Extract URL between < and >
                start = link_header.find("<")
                end = link_header.find(">")
                if start != -1 and end != -1:
                    next_url = link_header[start + 1 : end]

            time.sleep(0.1)  # Rate limiting

        except requests.RequestException as e:
            print(f"\n⚠️ Error: {e}")
            time.sleep(1)
            continue

    pbar.close()
    print(f"\n✓ Downloaded {len(proteins):,} proteins")

    # Save data
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as FASTA
    fasta_file = output_dir / "proteins.fasta"
    with open(fasta_file, "w") as f:
        for accession, data in proteins.items():
            gene = data.get("gene_name", "")
            header = f">{accession}|{gene}" if gene else f">{accession}"
            f.write(f"{header}\n{data['sequence']}\n")
    print(f"✓ Saved FASTA: {fasta_file}")

    # Save as JSON (with metadata)
    json_file = output_dir / "proteins.json"
    with open(json_file, "w") as f:
        json.dump(proteins, f, indent=2)
    print(f"✓ Saved JSON: {json_file}")

    # Save as TSV (summary)
    tsv_file = output_dir / "proteins.tsv"
    with open(tsv_file, "w") as f:
        f.write("accession\tgene_name\tprotein_name\tlength\torganism\n")
        for accession, data in proteins.items():
            f.write(
                f"{accession}\t{data['gene_name']}\t{data['protein_name']}\t"
                f"{data['length']}\t{data['organism']}\n"
            )
    print(f"✓ Saved TSV: {tsv_file}")

    return proteins


def download_proteins_by_ids(
    output_dir: Path,
    uniprot_ids: List[str],
) -> Optional[Dict[str, Dict]]:
    """
    Download specific proteins by UniProt IDs

    Args:
        output_dir: Output directory
        uniprot_ids: List of UniProt accessions or entry names

    Returns:
        Dictionary mapping UniProt IDs to protein data
    """
    print(f"Downloading {len(uniprot_ids)} specific proteins...")

    proteins: Dict[str, Dict] = {}

    # Query in batches
    batch_size = 100

    for i in tqdm(range(0, len(uniprot_ids), batch_size), desc="Fetching proteins"):
        batch = uniprot_ids[i : i + batch_size]
        query = " OR ".join([f"accession:{uid}" for uid in batch])

        params = {
            "query": query,
            "format": "json",
            "fields": "accession,id,gene_names,protein_name,sequence,length,organism_name",
            "size": batch_size,
        }

        try:
            response = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            for entry in data.get("results", []):
                accession = entry.get("primaryAccession", "")
                sequence = entry.get("sequence", {}).get("value", "")

                if accession and sequence:
                    gene_name = ""
                    genes = entry.get("genes", [])
                    if genes:
                        gene_name = genes[0].get("geneName", {}).get("value", "")

                    proteins[accession] = {
                        "accession": accession,
                        "gene_name": gene_name,
                        "sequence": sequence,
                        "length": len(sequence),
                    }

            time.sleep(0.2)

        except requests.RequestException as e:
            print(f"\n⚠️ Error: {e}")
            continue

    # Save data
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_file = output_dir / "proteins.fasta"
    with open(fasta_file, "w") as f:
        for accession, data in proteins.items():
            f.write(f">{accession}\n{data['sequence']}\n")

    print(f"✓ Downloaded {len(proteins)} proteins")
    return proteins


# =============================================================================
# ALPHAFOLD FUNCTIONS
# =============================================================================


def download_alphafold_structure(uniprot_id: str, output_dir: Path) -> Tuple[str, str]:
    """
    Download AlphaFold structure for a protein

    Args:
        uniprot_id: UniProt accession
        output_dir: Output directory

    Returns:
        Tuple of (status, uniprot_id)
    """
    url = f"{ALPHAFOLD_URL}/AF-{uniprot_id}-F1-model_v6.pdb"
    output_file = output_dir / f"{uniprot_id}.pdb"

    if output_file.exists():
        return ("skipped", uniprot_id)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            return ("downloaded", uniprot_id)
        else:
            return ("not_found", uniprot_id)
    except requests.RequestException:
        return ("not_found", uniprot_id)


def download_structures(
    proteins: Dict[str, Dict],
    output_dir: Path,
    max_workers: int = 10,
) -> Dict[str, int]:
    """
    Download AlphaFold structures for proteins

    Args:
        proteins: Dictionary of protein data
        output_dir: Output directory for structures
        max_workers: Number of parallel download threads

    Returns:
        Dictionary with download statistics
    """
    print()
    print("=" * 70)
    print("DOWNLOADING ALPHAFOLD STRUCTURES")
    print("=" * 70)
    print()

    uniprot_ids = list(proteins.keys())
    print(f"Total proteins: {len(uniprot_ids):,}")
    print(f"Parallel threads: {max_workers}")
    print()

    if not uniprot_ids:
        return {"downloaded": 0, "skipped": 0, "not_found": 0, "total": 0}

    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    not_found = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(download_alphafold_structure, uid, output_dir): uid
            for uid in uniprot_ids
        }

        with tqdm(total=len(uniprot_ids), desc="Downloading structures") as pbar:
            for future in as_completed(future_to_id):
                status, _ = future.result()

                if status == "downloaded":
                    downloaded += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    not_found += 1

                pbar.update(1)
                pbar.set_postfix(ok=downloaded, skip=skipped, miss=not_found)

    print()
    total = len(uniprot_ids)
    print(f"✓ Downloaded: {downloaded:,} ({downloaded / total * 100:.1f}%)")
    print(f"⊙ Skipped: {skipped:,} ({skipped / total * 100:.1f}%)")
    print(f"✗ Not found: {not_found:,} ({not_found / total * 100:.1f}%)")

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "not_found": not_found,
        "total": total,
    }


# =============================================================================
# STRING FUNCTIONS
# =============================================================================


def download_string_protein_info(
    taxon_id: int,
    output_dir: Path,
) -> Optional[Path]:
    """
    Download STRING protein information

    Args:
        taxon_id: NCBI taxonomy ID
        output_dir: Output directory

    Returns:
        Path to downloaded file
    """
    print()
    print("Downloading STRING protein info...")

    import gzip

    # Map strain-specific taxon IDs to STRING-compatible species IDs
    string_taxon_id = STRING_TAXON_ID_MAP.get(taxon_id, taxon_id)
    if string_taxon_id != taxon_id:
        print(f"  (mapped taxon {taxon_id} -> {string_taxon_id} for STRING)")

    filename = f"{string_taxon_id}.protein.info.v{STRING_VERSION}.txt.gz"
    url = f"{STRING_DOWNLOAD_URL}/protein.info.v{STRING_VERSION}/{filename}"

    output_file_gz = output_dir / filename
    output_file = output_dir / "string_protein_info.tsv"

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"⚠️ Could not download STRING info: {e}")
        return None

    total_size = int(response.headers.get("content-length", 0))

    with open(output_file_gz, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Downloading",
            disable=total_size == 0,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Decompress
    with gzip.open(output_file_gz, "rt") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            f_out.write(line)

    print(f"✓ Saved: {output_file}")
    return output_file


def download_string_sequences(
    taxon_id: int,
    output_dir: Path,
) -> Optional[Path]:
    """
    Download STRING protein sequences

    Args:
        taxon_id: NCBI taxonomy ID
        output_dir: Output directory

    Returns:
        Path to downloaded file
    """
    print()
    print("Downloading STRING sequences...")

    import gzip

    # Map strain-specific taxon IDs to STRING-compatible species IDs
    string_taxon_id = STRING_TAXON_ID_MAP.get(taxon_id, taxon_id)
    if string_taxon_id != taxon_id:
        print(f"  (mapped taxon {taxon_id} -> {string_taxon_id} for STRING)")

    filename = f"{string_taxon_id}.protein.sequences.v{STRING_VERSION}.fa.gz"
    url = f"{STRING_DOWNLOAD_URL}/protein.sequences.v{STRING_VERSION}/{filename}"

    output_file_gz = output_dir / filename
    output_file = output_dir / "string_sequences.fasta"

    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"⚠️ Could not download STRING sequences: {e}")
        return None

    total_size = int(response.headers.get("content-length", 0))

    with open(output_file_gz, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Downloading",
            disable=total_size == 0,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Decompress
    with gzip.open(output_file_gz, "rt") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            f_out.write(line)

    print(f"✓ Saved: {output_file}")
    return output_file


# =============================================================================
# ID MAPPING
# =============================================================================


def create_id_mapping(
    proteins: Dict[str, Dict],
    output_dir: Path,
) -> Dict[str, str]:
    """
    Create ID mapping file (gene name -> UniProt ID)

    Args:
        proteins: Dictionary of protein data
        output_dir: Output directory

    Returns:
        Dictionary mapping gene names to UniProt IDs
    """
    gene_to_uniprot: Dict[str, str] = {}
    uniprot_to_gene: Dict[str, str] = {}

    for accession, data in proteins.items():
        gene_name = data.get("gene_name", "")
        if gene_name:
            gene_to_uniprot[gene_name.upper()] = accession
            uniprot_to_gene[accession] = gene_name

    # Save mapping
    mapping_file = output_dir / "id_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(
            {
                "gene_to_uniprot": gene_to_uniprot,
                "uniprot_to_gene": uniprot_to_gene,
            },
            f,
            indent=2,
        )

    print(f"✓ Created ID mapping for {len(gene_to_uniprot):,} genes")
    return gene_to_uniprot


# =============================================================================
# METADATA AND SUMMARY
# =============================================================================


def save_metadata(
    output_dir: Path,
    taxon_id: int,
    species_name: str,
    filter_type: str,
    protein_count: int,
    include_structures: bool,
) -> None:
    """Save download metadata"""
    output_file = output_dir / "download_info.json"

    info = {
        "taxon_id": taxon_id,
        "species_name": species_name,
        "filter_type": filter_type,
        "protein_count": protein_count,
        "include_structures": include_structures,
        "sources": {
            "sequences": "UniProt",
            "structures": "AlphaFold" if include_structures else None,
        },
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_file, "w") as f:
        json.dump(info, f, indent=2)

    print(f"✓ Saved metadata: {output_file}")


def print_summary(output_dir: Path, proteins: Dict, include_structures: bool) -> None:
    """Print download summary"""
    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print(f"Proteins downloaded: {len(proteins):,}")
    print()
    print("Files created:")
    print("  • proteins.fasta - Protein sequences (FASTA)")
    print("  • proteins.json - Protein data with metadata (JSON)")
    print("  • proteins.tsv - Protein summary (TSV)")
    print("  • id_mapping.json - Gene name to UniProt ID mapping")
    if include_structures:
        print("  • structures/ - AlphaFold PDB structures")
    print("  • download_info.json - Download metadata")
    print()


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download Protein Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_pro.py --species "Homo sapiens"
    python download_pro.py --taxon-id 9606 --include-structures
    python download_pro.py --species "Homo sapiens" --filter dna-binding
    python download_pro.py --species "Homo sapiens" --filter rna-binding --max-proteins 1000

Available filters:
    all                 - All proteins (default)
    dna-binding         - DNA-binding proteins
    rna-binding         - RNA-binding proteins
    membrane            - Membrane proteins
    enzyme              - Enzymes
    kinase              - Kinases
    receptor            - Receptors
    transcription-factor - Transcription factors
    drug-target         - Known drug targets
        """,
    )

    parser.add_argument(
        "--species",
        type=str,
        help="Species name (e.g., 'Homo sapiens')",
    )

    parser.add_argument(
        "--taxon-id",
        type=int,
        help="NCBI Taxonomy ID (e.g., 9606 for human)",
    )

    parser.add_argument(
        "--uniprot-ids",
        type=str,
        help="Comma-separated list of UniProt IDs to download",
    )

    parser.add_argument(
        "--filter",
        type=str,
        default="all",
        choices=list(PROTEIN_FILTERS.keys()),
        help="Filter proteins by type (default: all)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/proteins",
        help="Output directory (default: data/proteins)",
    )

    parser.add_argument(
        "--max-proteins",
        type=int,
        help="Maximum number of proteins to download",
    )

    parser.add_argument(
        "--include-structures",
        action="store_true",
        help="Also download AlphaFold structures",
    )

    parser.add_argument(
        "--include-string",
        action="store_true",
        help="Also download STRING protein info",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=10,
        help="Number of parallel threads for structure download (default: 10)",
    )

    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Include unreviewed (TrEMBL) entries (default: only reviewed/Swiss-Prot)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.species and not args.taxon_id and not args.uniprot_ids:
        parser.print_help()
        print("\n❌ Error: Please provide --species, --taxon-id, or --uniprot-ids")
        return

    print("=" * 70)
    print("PROTEIN DATA DOWNLOAD")
    print("=" * 70)
    print()

    # Handle --include-unreviewed
    reviewed_only = not args.include_unreviewed

    # Determine output directory
    if args.species:
        species_dir = args.species.replace(" ", "_").lower()
    elif args.taxon_id:
        species_dir = f"taxon_{args.taxon_id}"
    else:
        species_dir = "custom"

    # Add filter to directory name if not "all"
    if args.filter != "all":
        species_dir = f"{species_dir}_{args.filter.replace('-', '_')}"

    output_dir = Path(args.output_dir) / species_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Download proteins
    if args.uniprot_ids:
        # Download specific proteins
        uniprot_ids = [uid.strip() for uid in args.uniprot_ids.split(",")]
        proteins = download_proteins_by_ids(output_dir, uniprot_ids)
        taxon_id = 0
        species_name = "Custom"
    else:
        # Get taxon ID
        taxon_id = args.taxon_id
        species_name = args.species or f"Taxon {taxon_id}"
        proteome_id = None

        if args.species and not taxon_id:
            proteome_data = search_proteome(args.species)
            if proteome_data:
                taxon_id = proteome_data.get("taxonomy", {}).get("taxonId")
                species_name = proteome_data.get("taxonomy", {}).get(
                    "scientificName", species_name
                )
                proteome_id = proteome_data.get("id")

        if not taxon_id:
            print("❌ Could not determine taxon ID")
            return

        print(f"Species: {species_name}")
        print(f"Taxon ID: {taxon_id}")
        print(f"Filter: {args.filter}")
        print()

        # Download proteins from UniProt
        proteins = download_proteins(
            output_dir=output_dir,
            taxon_id=taxon_id,
            filter_type=args.filter,
            reviewed_only=reviewed_only,
            proteome_id=proteome_id,
            max_proteins=args.max_proteins,
        )

    if not proteins:
        print("❌ No proteins downloaded")
        return

    # Create ID mapping
    create_id_mapping(proteins, output_dir)

    # Download AlphaFold structures if requested
    if args.include_structures:
        structures_dir = output_dir / "structures"
        download_structures(proteins, structures_dir, max_workers=args.threads)

    # Download STRING info if requested
    if args.include_string and taxon_id:
        string_dir = output_dir / "string"
        string_dir.mkdir(parents=True, exist_ok=True)
        download_string_protein_info(taxon_id, string_dir)
        download_string_sequences(taxon_id, string_dir)

    # Save metadata
    save_metadata(
        output_dir=output_dir,
        taxon_id=taxon_id,
        species_name=species_name,
        filter_type=args.filter,
        protein_count=len(proteins),
        include_structures=args.include_structures,
    )

    # Print summary
    print_summary(output_dir, proteins, args.include_structures)

    print("Done!")
    print()
    print("This protein data can be used with:")
    print("  • download_p2p.py - Protein-protein interactions")
    print("  • download_p2d.py - Protein-DNA interactions")
    print("  • download_p2r.py - Protein-RNA interactions")
    print("  • download_p2m.py - Protein-molecule interactions")


if __name__ == "__main__":
    main()
