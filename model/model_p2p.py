#!/usr/bin/env python3
"""
Protein-Protein Interaction Prediction Model
=============================================

Multi-modal model combining:
1. ESM-2 pretrained protein language model for sequence encoding
2. AlphaFold structure encoding via contact maps
3. Graph Neural Network for PPI network topology

Expects pre-processed data from prepare/ scripts.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import esm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import PDBParser
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

# Constants
CONTACT_THRESHOLD = 8.0  # Angstroms for contact map


class StructureParser:
    """Parse AlphaFold PDB structures and extract contact maps"""

    def __init__(self, contact_threshold: float = CONTACT_THRESHOLD):
        self.contact_threshold = contact_threshold
        self.parser = PDBParser(QUIET=True)
        self.cache = {}

    def parse_pdb(self, pdb_path: Path) -> Dict:
        """Parse PDB file and extract structural features"""
        if str(pdb_path) in self.cache:
            return self.cache[str(pdb_path)]

        if not pdb_path.exists():
            raise FileNotFoundError(f"Structure file not found: {pdb_path}")

        structure = self.parser.get_structure("protein", str(pdb_path))
        model = structure[0]
        chains = list(model.get_chains())
        if not chains:
            raise ValueError(f"No chains found in structure: {pdb_path}")

        chain = chains[0]
        ca_coords = []

        for residue in chain.get_residues():
            if residue.id[0] != " " or "CA" not in residue:
                continue
            ca_coords.append(residue["CA"].get_coord())

        if len(ca_coords) < 10:
            raise ValueError(
                f"Too few CA atoms ({len(ca_coords)}) in structure: {pdb_path}"
            )

        ca_coords = np.array(ca_coords, dtype=np.float32)

        # Compute distance matrix and contact map
        diff = ca_coords[:, None, :] - ca_coords[None, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1))
        contact_map = (distances < self.contact_threshold).astype(np.float32)

        result = {"contact_map": contact_map, "length": len(ca_coords)}
        self.cache[str(pdb_path)] = result
        return result

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


class PPIGraph:
    """
    Protein-protein interaction graph for GNN encoding.
    Expects pre-processed data from prepare/ scripts.
    """

    def __init__(
        self,
        protein_ids: List[str],
        edge_index: torch.Tensor,
        sequences: Dict[str, str],
        device: torch.device,
    ):
        self.device = device
        self.sequences = sequences

        self.uniprot_to_idx = {uid: idx for idx, uid in enumerate(protein_ids)}
        self.idx_to_uniprot = {idx: uid for idx, uid in enumerate(protein_ids)}
        self.num_proteins = len(protein_ids)

        self.edge_index = edge_index.to(device)
        self.gnn_embeddings: Optional[torch.Tensor] = None

    def compute_gnn_embeddings(
        self,
        gnn_encoder: nn.Module,
        sequence_encoder: nn.Module,
        batch_size: int = 32,
    ) -> None:
        """Pre-compute GNN embeddings for all proteins"""
        all_embeddings = []
        protein_ids = [self.idx_to_uniprot[i] for i in range(self.num_proteins)]

        with torch.no_grad():
            for i in range(0, len(protein_ids), batch_size):
                batch_ids = protein_ids[i : i + batch_size]
                batch_seqs = [self.sequences[uid] for uid in batch_ids]
                embeddings = sequence_encoder(batch_seqs)
                all_embeddings.append(embeddings.cpu())

        node_features = torch.cat(all_embeddings, dim=0).to(self.device)

        gnn_encoder.eval()
        with torch.no_grad():
            self.gnn_embeddings = gnn_encoder(node_features, self.edge_index)
        gnn_encoder.train()

    def get_embeddings(self, uniprot_ids: List[str]) -> torch.Tensor:
        """Get pre-computed GNN embeddings for a list of proteins"""
        indices = [self.uniprot_to_idx[uid] for uid in uniprot_ids]
        return self.gnn_embeddings[indices]


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
        contact_maps: torch.Tensor,
        gnn_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Encode proteins using all modalities

        All inputs are required - no fallback to zeros.
        """
        # Sequence encoding (ESM-2)
        seq_emb = self.sequence_encoder(sequences)

        # Structure encoding
        struct_emb = self.structure_encoder(contact_maps.to(self.device))

        # GNN encoding (pre-computed node embeddings)
        gnn_emb = gnn_embeddings.to(self.device)

        # Concatenate all modalities
        combined = torch.cat([seq_emb, struct_emb, gnn_emb], dim=-1)
        return combined

    def forward(
        self,
        seq1: List[str],
        seq2: List[str],
        contact1: torch.Tensor,
        contact2: torch.Tensor,
        gnn_emb1: torch.Tensor,
        gnn_emb2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict protein-protein interaction

        All inputs are required - no fallback to zeros.
        """
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
    """Dataset for protein-protein interactions. Expects pre-processed data from prepare/ scripts."""

    def __init__(
        self,
        data_file: Path,
        sequences: Dict[str, str],
        structures_dir: Path,
        ppi_graph: PPIGraph,
        max_struct_length: int = 500,
    ):
        self.sequences = sequences
        self.structures_dir = structures_dir
        self.ppi_graph = ppi_graph
        self.max_struct_length = max_struct_length
        self.structure_parser = StructureParser()

        self.samples = []
        with open(data_file, "r") as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                u1 = parts[0]
                u2 = parts[1]
                label = float(parts[2])
                score = float(parts[3])
                self.samples.append((u1, u2, label, score))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u1, u2, label, score = self.samples[idx]

        pdb1 = self.structures_dir / f"{u1}.pdb"
        pdb2 = self.structures_dir / f"{u2}.pdb"

        parsed1 = self.structure_parser.parse_pdb(pdb1)
        parsed2 = self.structure_parser.parse_pdb(pdb2)

        return {
            "uniprot1": u1,
            "uniprot2": u2,
            "seq1": self.sequences[u1],
            "seq2": self.sequences[u2],
            "label": torch.tensor(label, dtype=torch.float32),
            "score": torch.tensor(score, dtype=torch.float32),
            "contact1": self.structure_parser.get_contact_tensor(
                parsed1["contact_map"], self.max_struct_length
            ),
            "contact2": self.structure_parser.get_contact_tensor(
                parsed2["contact_map"], self.max_struct_length
            ),
        }


def collate_fn(batch, ppi_graph: PPIGraph):
    """Collate function for DataLoader"""
    uniprot1_list = [b["uniprot1"] for b in batch]
    uniprot2_list = [b["uniprot2"] for b in batch]

    return {
        "seq1": [b["seq1"] for b in batch],
        "seq2": [b["seq2"] for b in batch],
        "contact1": torch.stack([b["contact1"] for b in batch]),
        "contact2": torch.stack([b["contact2"] for b in batch]),
        "gnn_emb1": ppi_graph.get_embeddings(uniprot1_list),
        "gnn_emb2": ppi_graph.get_embeddings(uniprot2_list),
        "label": torch.stack([b["label"] for b in batch]),
        "score": torch.stack([b["score"] for b in batch]),
    }


def load_sequences(fasta_file: Path) -> Dict[str, str]:
    """Load protein sequences from FASTA file"""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = "".join(current_seq)

    return sequences


def load_graph(graph_file: Path) -> Tuple[List[str], torch.Tensor]:
    """Load pre-processed graph from file"""
    data = torch.load(graph_file)
    return data["protein_ids"], data["edge_index"]


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
            gnn_emb1=batch["gnn_emb1"],
            gnn_emb2=batch["gnn_emb2"],
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
                gnn_emb1=batch["gnn_emb1"],
                gnn_emb2=batch["gnn_emb2"],
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

    return history
