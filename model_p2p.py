#!/usr/bin/env python3
"""
Protein-Protein Interaction Prediction Model V2
================================================

Multi-modal model combining:
1. ESM-2 pretrained protein language model for sequence encoding
2. AlphaFold structure encoding via contact maps
3. Graph Neural Network for PPI network topology

Uses existing proteome data from download_proteome.py and maps STRING IDs to UniProt IDs.

Usage:
    python model_v2.py --data-dir data/taxon_9606 --proteome-dir data/up000005640 --epochs 50
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import esm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import PDBParser
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

# Constants
CONTACT_THRESHOLD = 8.0  # Angstroms for contact map


class IDMapper:
    """Map between STRING protein IDs and UniProt IDs using gene names"""

    def __init__(self, string_info_file: Path, uniprot_fasta: Path):
        self.string_to_gene = {}  # STRING ID -> gene name
        self.gene_to_uniprot = {}  # gene name -> UniProt ID
        self.string_to_uniprot = {}  # STRING ID -> UniProt ID (direct mapping)

        # Load STRING protein info (STRING ID -> gene name)
        if string_info_file.exists():
            print(f"Loading STRING protein info...")
            with open(string_info_file, "r") as f:
                f.readline()  # Skip header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        string_id = parts[0]
                        gene_name = parts[1].upper()
                        self.string_to_gene[string_id] = gene_name

        # Load UniProt FASTA (gene name -> UniProt ID)
        if uniprot_fasta.exists():
            print(f"Loading UniProt gene mappings...")
            with open(uniprot_fasta, "r") as f:
                for line in f:
                    if line.startswith(">"):
                        # Parse: >tr|A0A087WVL8|A0A087WVL8_HUMAN ... GN=FMR1 ...
                        parts = line.split("|")
                        if len(parts) >= 2:
                            uniprot_id = parts[1]
                            # Extract gene name
                            if "GN=" in line:
                                gn_start = line.index("GN=") + 3
                                gn_end = line.find(" ", gn_start)
                                if gn_end == -1:
                                    gn_end = len(line)
                                gene_name = line[gn_start:gn_end].strip().upper()
                                self.gene_to_uniprot[gene_name] = uniprot_id

        # Build STRING -> UniProt mapping
        for string_id, gene_name in self.string_to_gene.items():
            if gene_name in self.gene_to_uniprot:
                self.string_to_uniprot[string_id] = self.gene_to_uniprot[gene_name]

        print(f"  ✓ Mapped {len(self.string_to_uniprot):,} STRING IDs to UniProt IDs")

    def get_uniprot_id(self, string_id: str) -> Optional[str]:
        return self.string_to_uniprot.get(string_id)


class StructureParser:
    """Parse AlphaFold PDB structures and extract contact maps"""

    def __init__(self, contact_threshold: float = CONTACT_THRESHOLD):
        self.contact_threshold = contact_threshold
        self.parser = PDBParser(QUIET=True)
        self.cache = {}

    def parse_pdb(self, pdb_path: Path) -> Optional[Dict]:
        """Parse PDB file and extract structural features"""
        if str(pdb_path) in self.cache:
            return self.cache[str(pdb_path)]

        try:
            structure = self.parser.get_structure("protein", str(pdb_path))
            model = structure[0]
            chains = list(model.get_chains())
            if not chains:
                return None

            chain = chains[0]
            ca_coords = []

            for residue in chain.get_residues():
                if residue.id[0] != " " or "CA" not in residue:
                    continue
                ca_coords.append(residue["CA"].get_coord())

            if len(ca_coords) < 10:
                return None

            ca_coords = np.array(ca_coords, dtype=np.float32)

            # Compute distance matrix and contact map
            diff = ca_coords[:, None, :] - ca_coords[None, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=-1))
            contact_map = (distances < self.contact_threshold).astype(np.float32)

            result = {"contact_map": contact_map, "length": len(ca_coords)}
            self.cache[str(pdb_path)] = result
            return result
        except Exception:
            return None

    def get_contact_tensor(
        self, contact_map: np.ndarray, max_length: int = 500
    ) -> torch.Tensor:
        """Convert contact map to fixed-size tensor"""
        L = contact_map.shape[0]

        if L > max_length:
            step = L / max_length
            indices = [int(i * step) for i in range(max_length)]
            contact_map = contact_map[np.ix_(indices, indices)]
        elif L < max_length:
            padded = np.zeros((max_length, max_length), dtype=np.float32)
            padded[:L, :L] = contact_map
            contact_map = padded

        return torch.tensor(contact_map, dtype=torch.float32)


class ESMEncoder(nn.Module):
    """ESM-2 protein sequence encoder"""

    def __init__(
        self,
        model_name: str = "esm2_t12_35M_UR50D",
        output_dim: int = 512,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.output_dim = output_dim

        # Load ESM-2 model
        print(f"Loading ESM-2 model: {model_name}")
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            model_name
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_dim = self.esm_model.embed_dim

        # Freeze ESM weights
        for param in self.esm_model.parameters():
            param.requires_grad = False

        self.esm_model = self.esm_model.to(self.device)
        self.esm_model.eval()

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.esm_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Encode sequences using ESM-2"""
        # Truncate sequences to ESM max length
        data = [(f"p{i}", seq[:1022]) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.esm_model(
                batch_tokens, repr_layers=[self.esm_model.num_layers]
            )

        token_repr = results["representations"][self.esm_model.num_layers]

        # Mean pooling over sequence (excluding special tokens)
        embeddings = []
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), 1022)
            emb = token_repr[i, 1 : seq_len + 1].mean(dim=0)
            embeddings.append(emb)

        embeddings = torch.stack(embeddings)
        return self.projection(embeddings)


class StructureEncoder(nn.Module):
    """Encode protein structure from contact maps using CNN"""

    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.output_dim = output_dim

        # CNN with strided convolutions (MPS compatible - no adaptive pooling)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # After CNN, we use global average pooling which gives 128 features
        self.projection = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, contact_maps: torch.Tensor) -> torch.Tensor:
        """Encode contact maps"""
        x = contact_maps.unsqueeze(1)  # [B, 1, L, L]
        x = self.cnn(x)
        x = x.mean(dim=[2, 3])  # Global average pooling -> [B, 128]
        return self.projection(x)


class GNNEncoder(nn.Module):
    """Graph Neural Network for PPI network encoding"""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(SAGEConv(in_dim, out_dim))
            if i < num_layers - 1:
                self.norms.append(nn.LayerNorm(out_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GNN layers"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.norms):
                x = self.norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
        return x


class PPIModel(nn.Module):
    """
    Multi-modal Protein-Protein Interaction Predictor

    Combines ESM-2 sequence embeddings, structure features, and GNN network embeddings
    """

    def __init__(
        self,
        protein_dim: int = 512,
        hidden_dim: int = 512,
        esm_model: str = "esm2_t12_35M_UR50D",
        gnn_layers: int = 3,
        dropout: float = 0.1,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.protein_dim = protein_dim

        # ESM-2 sequence encoder
        self.sequence_encoder = ESMEncoder(
            model_name=esm_model, output_dim=protein_dim, device=self.device
        )

        # Structure encoder
        self.structure_encoder = StructureEncoder(output_dim=protein_dim)

        # GNN encoder
        self.gnn_encoder = GNNEncoder(
            input_dim=protein_dim,
            hidden_dim=hidden_dim,
            output_dim=protein_dim,
            num_layers=gnn_layers,
            dropout=dropout,
        )

        # Modality fusion - combines sequence, structure, and GNN embeddings
        # For each protein pair: concat(p1, p2) + hadamard(p1, p2) + |p1 - p2|
        # With 3 modalities per protein = 3 * protein_dim per protein
        fusion_input = protein_dim * 3 * 4  # 3 modalities, 4 combination types

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Prediction heads
        self.interaction_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def encode_proteins(
        self,
        sequences: List[str],
        contact_maps: torch.Tensor = None,
        gnn_embeddings: torch.Tensor = None,
    ) -> torch.Tensor:
        """Encode proteins using all modalities"""
        batch_size = len(sequences)

        # Sequence encoding (ESM-2)
        seq_emb = self.sequence_encoder(sequences)

        # Structure encoding
        if contact_maps is not None:
            struct_emb = self.structure_encoder(contact_maps.to(self.device))
        else:
            struct_emb = torch.zeros(batch_size, self.protein_dim, device=self.device)

        # GNN encoding (pre-computed node embeddings)
        if gnn_embeddings is not None:
            gnn_emb = gnn_embeddings.to(self.device)
        else:
            gnn_emb = torch.zeros(batch_size, self.protein_dim, device=self.device)

        # Concatenate all modalities
        combined = torch.cat([seq_emb, struct_emb, gnn_emb], dim=-1)
        return combined

    def forward(
        self,
        seq1: List[str],
        seq2: List[str],
        contact1: torch.Tensor = None,
        contact2: torch.Tensor = None,
        gnn_emb1: torch.Tensor = None,
        gnn_emb2: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict protein-protein interaction"""
        # Encode both proteins
        emb1 = self.encode_proteins(seq1, contact1, gnn_emb1)
        emb2 = self.encode_proteins(seq2, contact2, gnn_emb2)

        # Combine pair embeddings
        concat = torch.cat([emb1, emb2], dim=-1)
        product = emb1 * emb2
        diff = torch.abs(emb1 - emb2)
        combined = torch.cat([concat, product, diff], dim=-1)

        # Fusion and prediction
        fused = self.fusion(combined)
        interaction_logits = self.interaction_head(fused).squeeze(-1)
        confidence_scores = self.confidence_head(fused).squeeze(-1)

        return interaction_logits, confidence_scores


class PPIDataset(Dataset):
    """Dataset for protein-protein interactions"""

    def __init__(
        self,
        interactions: List[Tuple],
        sequences: Dict[str, str],
        structures_dir: Path,
        id_mapper: IDMapper,
        max_struct_length: int = 500,
        negative_ratio: float = 1.0,
        seed: int = 42,
    ):
        self.sequences = sequences
        self.structures_dir = structures_dir
        self.id_mapper = id_mapper
        self.max_struct_length = max_struct_length
        self.structure_parser = StructureParser()

        random.seed(seed)
        np.random.seed(seed)

        # Filter to proteins with sequences (via UniProt mapping)
        print("Building dataset...")
        self.positive_pairs = []
        self.scores = []

        for p1, p2, score in interactions:
            uniprot1 = id_mapper.get_uniprot_id(p1)
            uniprot2 = id_mapper.get_uniprot_id(p2)
            if (
                uniprot1
                and uniprot2
                and uniprot1 in sequences
                and uniprot2 in sequences
            ):
                self.positive_pairs.append((p1, p2, uniprot1, uniprot2))
                self.scores.append(score / 1000.0)

        print(f"  ✓ {len(self.positive_pairs):,} positive pairs with sequences")

        # Count pairs with structures
        struct_count = 0
        for _, _, u1, u2 in self.positive_pairs:
            if (self.structures_dir / f"{u1}.pdb").exists() and (
                self.structures_dir / f"{u2}.pdb"
            ).exists():
                struct_count += 1
        print(f"  ✓ {struct_count:,} pairs with both structures")

        # Track positive pairs for negative sampling
        self.positive_set = set()
        for p1, p2, _, _ in self.positive_pairs:
            self.positive_set.add((p1, p2))
            self.positive_set.add((p2, p1))

        # Get all STRING IDs that have UniProt mappings
        self.protein_list = list(
            set(p for p1, p2, _, _ in self.positive_pairs for p in [p1, p2])
        )

        # Generate negative pairs
        num_neg = int(len(self.positive_pairs) * negative_ratio)
        self.negative_pairs = []
        attempts = 0

        while len(self.negative_pairs) < num_neg and attempts < num_neg * 10:
            p1 = random.choice(self.protein_list)
            p2 = random.choice(self.protein_list)
            if p1 != p2 and (p1, p2) not in self.positive_set:
                u1 = id_mapper.get_uniprot_id(p1)
                u2 = id_mapper.get_uniprot_id(p2)
                if u1 and u2:
                    self.negative_pairs.append((p1, p2, u1, u2))
                    self.positive_set.add((p1, p2))
                    self.positive_set.add((p2, p1))
            attempts += 1

        print(f"  ✓ {len(self.negative_pairs):,} negative pairs")

        # Combine samples
        self.samples = []
        for (p1, p2, u1, u2), score in zip(self.positive_pairs, self.scores):
            self.samples.append((u1, u2, 1.0, score))
        for p1, p2, u1, u2 in self.negative_pairs:
            self.samples.append((u1, u2, 0.0, 0.0))

        random.shuffle(self.samples)
        print(f"  ✓ {len(self.samples):,} total samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u1, u2, label, score = self.samples[idx]

        result = {
            "uniprot1": u1,
            "uniprot2": u2,
            "seq1": self.sequences[u1],
            "seq2": self.sequences[u2],
            "label": torch.tensor(label, dtype=torch.float32),
            "score": torch.tensor(score, dtype=torch.float32),
        }

        # Load structure features
        pdb1 = self.structures_dir / f"{u1}.pdb"
        pdb2 = self.structures_dir / f"{u2}.pdb"

        if pdb1.exists():
            parsed = self.structure_parser.parse_pdb(pdb1)
            if parsed:
                result["contact1"] = self.structure_parser.get_contact_tensor(
                    parsed["contact_map"], self.max_struct_length
                )
            else:
                result["contact1"] = torch.zeros(
                    self.max_struct_length, self.max_struct_length
                )
        else:
            result["contact1"] = torch.zeros(
                self.max_struct_length, self.max_struct_length
            )

        if pdb2.exists():
            parsed = self.structure_parser.parse_pdb(pdb2)
            if parsed:
                result["contact2"] = self.structure_parser.get_contact_tensor(
                    parsed["contact_map"], self.max_struct_length
                )
            else:
                result["contact2"] = torch.zeros(
                    self.max_struct_length, self.max_struct_length
                )
        else:
            result["contact2"] = torch.zeros(
                self.max_struct_length, self.max_struct_length
            )

        return result


def collate_fn(batch):
    """Collate function for DataLoader"""
    return {
        "uniprot1": [b["uniprot1"] for b in batch],
        "uniprot2": [b["uniprot2"] for b in batch],
        "seq1": [b["seq1"] for b in batch],
        "seq2": [b["seq2"] for b in batch],
        "contact1": torch.stack([b["contact1"] for b in batch]),
        "contact2": torch.stack([b["contact2"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "score": torch.stack([b["score"] for b in batch]),
    }


class DataManager:
    """Manage data loading for STRING + UniProt proteome"""

    def __init__(self, string_dir: Path, proteome_dir: Path):
        self.string_dir = Path(string_dir)
        self.proteome_dir = Path(proteome_dir)
        self.structures_dir = self.proteome_dir / "structures"

        # Create ID mapper
        string_info = self.string_dir / "string" / "string_protein_info.tsv"
        uniprot_fasta = self.proteome_dir / "proteins.fasta"
        self.id_mapper = IDMapper(string_info, uniprot_fasta)

    def load_interactions(self, min_score: int = 0, max_count: int = None) -> List:
        """Load protein interactions from STRING"""
        interactions_file = self.string_dir / "string" / "string_interactions.tsv"
        if not interactions_file.exists():
            raise FileNotFoundError(f"Not found: {interactions_file}")

        print(f"Loading interactions from: {interactions_file}")

        interactions = []
        with open(interactions_file, "r") as f:
            f.readline()  # Skip header
            for line in f:
                if max_count and len(interactions) >= max_count:
                    break
                parts = line.strip().split()
                if len(parts) >= 3:
                    p1, p2, score = parts[0], parts[1], int(parts[2])
                    if score >= min_score:
                        interactions.append((p1, p2, score))

        print(f"  ✓ Loaded {len(interactions):,} interactions")
        return interactions

    def load_sequences(self) -> Dict[str, str]:
        """Load protein sequences from UniProt proteome"""
        fasta_file = self.proteome_dir / "proteins.fasta"
        if not fasta_file.exists():
            raise FileNotFoundError(f"Not found: {fasta_file}")

        print(f"Loading sequences from: {fasta_file}")

        sequences = {}
        current_id = None
        current_seq = []

        with open(fasta_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        sequences[current_id] = "".join(current_seq)
                    # Parse UniProt ID from header: >tr|A0A087WVL8|...
                    parts = line.split("|")
                    if len(parts) >= 2:
                        current_id = parts[1]
                    else:
                        current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id:
                sequences[current_id] = "".join(current_seq)

        print(f"  ✓ Loaded {len(sequences):,} sequences")
        return sequences


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        labels = batch["label"].to(device)
        scores = batch["score"].to(device)

        optimizer.zero_grad()

        logits, pred_scores = model(
            seq1=batch["seq1"],
            seq2=batch["seq2"],
            contact1=batch["contact1"],
            contact2=batch["contact2"],
        )

        # Combined loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        pos_mask = labels > 0.5
        if pos_mask.sum() > 0:
            score_loss = F.mse_loss(pred_scores[pos_mask], scores[pos_mask])
        else:
            score_loss = 0.0

        loss = bce_loss + 0.5 * score_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / len(dataloader), "accuracy": correct / total}


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_labels, all_probs = [], []
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            labels = batch["label"].to(device)

            logits, _ = model(
                seq1=batch["seq1"],
                seq2=batch["seq2"],
                contact1=batch["contact1"],
                contact2=batch["contact2"],
            )

            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    auc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs > 0.5).astype(float)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary", zero_division=0
    )

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_model(
    model, train_loader, val_loader, epochs, lr, device, save_path, patience=10
):
    """Full training loop"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_auc = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_metrics["auc"])

        print(
            f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f}"
        )
        print(
            f"  Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f}"
        )
        print(
            f"  Val AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f} | "
            f"P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f}"
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_f1"].append(val_metrics["f1"])

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "history": history,
                },
                save_path,
            )
            print(f"  ✓ Saved best model (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print("\n" + "=" * 70)
    print(f"Training complete. Best AUC: {best_auc:.4f}")
    print("=" * 70)

    return history


def main():
    parser = argparse.ArgumentParser(description="Train PPI Prediction Model V2")
    parser.add_argument(
        "--string-dir",
        type=str,
        required=True,
        help="Directory containing STRING data (e.g., data/taxon_9606)",
    )
    parser.add_argument(
        "--proteome-dir",
        type=str,
        required=True,
        help="Directory containing UniProt proteome (e.g., data/up000005640)",
    )
    parser.add_argument("--checkpoint", type=str, default="ppi_model_v2.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-score", type=int, default=0)
    parser.add_argument("--max-interactions", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--esm-model",
        type=str,
        default="esm2_t12_35M_UR50D",
        help="ESM model: esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D",
    )

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("PROTEIN-PROTEIN INTERACTION PREDICTION MODEL V2")
    print("=" * 70)
    print(f"\nSTRING data: {args.string_dir}")
    print(f"Proteome: {args.proteome_dir}")
    print(f"Device: {device}")
    print(f"ESM Model: {args.esm_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print()

    # Load data
    dm = DataManager(Path(args.string_dir), Path(args.proteome_dir))

    # Load interactions
    interactions = dm.load_interactions(
        min_score=args.min_score, max_count=args.max_interactions
    )

    # Load sequences
    sequences = dm.load_sequences()
    if not sequences:
        print("❌ No sequences available")
        return

    print()
    print("=" * 70)
    print("CREATING DATASET")
    print("=" * 70)

    dataset = PPIDataset(
        interactions=interactions,
        sequences=sequences,
        structures_dir=dm.structures_dir,
        id_mapper=dm.id_mapper,
        negative_ratio=1.0,
        seed=args.seed,
    )

    if len(dataset) == 0:
        print("❌ No valid samples in dataset")
        return

    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"\nTrain: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Create model
    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    model = PPIModel(
        protein_dim=512,
        hidden_dim=512,
        esm_model=args.esm_model,
        gnn_layers=3,
        dropout=0.1,
        device=device,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train
    save_path = Path(args.checkpoint)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=save_path,
        patience=10,
    )

    # Save history
    with open(save_path.with_suffix(".history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
