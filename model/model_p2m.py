#!/usr/bin/env python3
"""
Protein-Small Molecule Interaction Prediction Model
====================================================

Multi-modal model for predicting protein-ligand interactions:
1. ESM-2 for protein sequence encoding
2. Graph Neural Network for molecular structure encoding (SMILES → Graph)
3. Cross-attention fusion between protein and molecule representations

This model predicts drug-target interactions and binding affinities,
useful for drug discovery, virtual screening, and lead optimization.

Usage:
    python model_p2m.py --data-dir data/protein_molecule --epochs 50
    python model_p2m.py --data-dir data/protein_molecule --epochs 10 --max-samples 5000

Data Sources:
    - ChEMBL: Bioactivity data for drug-like molecules
    - BindingDB: Protein-ligand binding affinities
    - PDBbind: Experimentally measured binding data
    - DrugBank: Drug-target interactions
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Enable MPS fallback for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import esm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

# Atom and bond vocabulary for molecular graphs
ATOM_VOCAB = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "F": 4,
    "Cl": 5,
    "Br": 6,
    "I": 7,
    "P": 8,
    "B": 9,
    "Si": 10,
    "Se": 11,
    "H": 12,
    "UNK": 13,
}
ATOM_VOCAB_SIZE = 14

BOND_VOCAB = {
    "SINGLE": 0,
    "DOUBLE": 1,
    "TRIPLE": 2,
    "AROMATIC": 3,
}
BOND_VOCAB_SIZE = 4

# SMILES character vocabulary for sequence-based encoding
SMILES_VOCAB = {
    "PAD": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "S": 4,
    "F": 5,
    "Cl": 6,
    "Br": 7,
    "I": 8,
    "P": 9,
    "(": 10,
    ")": 11,
    "[": 12,
    "]": 13,
    "=": 14,
    "#": 15,
    "@": 16,
    "+": 17,
    "-": 18,
    "\\": 19,
    "/": 20,
    "1": 21,
    "2": 22,
    "3": 23,
    "4": 24,
    "5": 25,
    "6": 26,
    "7": 27,
    "8": 28,
    "9": 29,
    "0": 30,
    "c": 31,
    "n": 32,
    "o": 33,
    "s": 34,
    "H": 35,
    ".": 36,
    "UNK": 37,
}
SMILES_VOCAB_SIZE = 38
SMILES_PAD_IDX = 0


def encode_smiles(smiles: str, max_length: int = 150) -> torch.Tensor:
    """
    Encode SMILES string to tensor using character-level encoding

    Args:
        smiles: SMILES string representation of molecule
        max_length: Maximum sequence length

    Returns:
        Tensor of shape [max_length] with token indices
    """
    indices = []
    i = 0
    while i < len(smiles) and len(indices) < max_length:
        # Handle two-character tokens (Cl, Br)
        if i < len(smiles) - 1 and smiles[i : i + 2] in SMILES_VOCAB:
            indices.append(SMILES_VOCAB[smiles[i : i + 2]])
            i += 2
        elif smiles[i] in SMILES_VOCAB:
            indices.append(SMILES_VOCAB[smiles[i]])
            i += 1
        else:
            indices.append(SMILES_VOCAB["UNK"])
            i += 1

    # Pad to max_length
    if len(indices) < max_length:
        indices.extend([SMILES_PAD_IDX] * (max_length - len(indices)))

    return torch.tensor(indices, dtype=torch.long)


class MoleculeEncoder(nn.Module):
    """
    Small molecule encoder using CNN + Self-Attention on SMILES

    This is a sequence-based approach that works directly on SMILES strings,
    avoiding the need for RDKit dependency while still capturing molecular structure.

    Architecture:
    - Character embedding for SMILES tokens
    - Multi-scale CNN for local substructure patterns
    - Self-attention for long-range dependencies (ring systems, etc.)
    - Pooling with learned attention weights
    """

    def __init__(
        self,
        vocab_size: int = SMILES_VOCAB_SIZE,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=SMILES_PAD_IDX)

        # Positional encoding
        self.pos_embedding = nn.Embedding(500, embed_dim)  # Max 500 characters

        # Multi-scale CNN for capturing different substructure sizes
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim, hidden_dim // 4, kernel_size=k, padding=k // 2
                    ),
                    nn.BatchNorm1d(hidden_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for k in [3, 5, 7, 9]  # Different substructure sizes
            ]
        )

        # Combine multi-scale features
        self.conv_combine = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Transformer encoder layers for global context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode molecule from SMILES tokens

        Args:
            x: SMILES token indices [batch_size, seq_len]

        Returns:
            Tuple of (pooled_output, sequence_output)
            - pooled_output: [batch_size, output_dim]
            - sequence_output: [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len = x.shape

        # Create padding mask
        padding_mask = x == SMILES_PAD_IDX  # [batch, seq_len]

        # Embed characters with positional encoding
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        embedded = self.embedding(x) + self.pos_embedding(positions)

        # CNN expects [batch, channels, seq_len]
        conv_input = embedded.transpose(1, 2)

        # Multi-scale CNN
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(conv_input)
            # Ensure same sequence length
            if conv_out.shape[2] != seq_len:
                conv_out = F.pad(conv_out, (0, seq_len - conv_out.shape[2]))
            conv_outputs.append(conv_out)

        # Concatenate multi-scale features
        conv_combined = torch.cat(conv_outputs, dim=1)  # [batch, hidden_dim, seq_len]
        conv_combined = self.conv_combine(conv_combined)
        conv_out = conv_combined.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        # Create attention mask for transformer (True = ignore)
        # Note: for MPS compatibility, we handle masking differently
        transformer_out = self.transformer(conv_out)

        # Zero out padded positions
        mask = (~padding_mask).float().unsqueeze(-1)  # [batch, seq_len, 1]
        transformer_out = transformer_out * mask

        # Project to output dimension
        sequence_output = self.projection(transformer_out)

        # Attention-weighted pooling
        attn_weights = self.attention_pool(transformer_out)
        attn_weights = attn_weights.masked_fill(
            padding_mask.unsqueeze(-1), float("-inf")
        )
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled_output = (sequence_output * attn_weights).sum(dim=1)

        return pooled_output, sequence_output


class ProteinEncoder(nn.Module):
    """ESM-2 based protein encoder optimized for binding site prediction"""

    def __init__(
        self,
        model_name: str = "esm2_t12_35M_UR50D",
        output_dim: int = 512,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.output_dim = output_dim

        # Load ESM-2
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

        # Projection
        self.projection = nn.Sequential(
            nn.Linear(self.esm_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )

        # Binding pocket attention
        # Helps focus on residues likely to be in binding pockets
        self.pocket_attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.Tanh(),
            nn.Linear(output_dim // 4, 1),
        )

    def forward(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode protein sequences

        Returns:
            Tuple of (pooled_output, sequence_output)
        """
        # Truncate to ESM max length
        data = [(f"p{i}", seq[:1022]) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.esm_model(
                batch_tokens,
                repr_layers=[self.esm_model.num_layers],
                return_contacts=False,
            )

        token_repr = results["representations"][self.esm_model.num_layers]

        # Get sequence representations (excluding BOS/EOS)
        sequence_outputs = []
        pooled_outputs = []

        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), 1022)
            seq_repr = token_repr[i, 1 : seq_len + 1]  # Exclude BOS token
            sequence_outputs.append(seq_repr)
            pooled_outputs.append(seq_repr.mean(dim=0))

        # Pad sequence outputs to same length
        max_len = max(s.shape[0] for s in sequence_outputs)
        padded_outputs = []
        for seq_out in sequence_outputs:
            if seq_out.shape[0] < max_len:
                padding = torch.zeros(
                    max_len - seq_out.shape[0], seq_out.shape[1], device=seq_out.device
                )
                seq_out = torch.cat([seq_out, padding], dim=0)
            padded_outputs.append(seq_out)

        sequence_output = torch.stack(padded_outputs)
        pooled_output = torch.stack(pooled_outputs)

        # Project
        pooled_output = self.projection(pooled_output)
        sequence_output = self.projection(sequence_output)

        # Apply binding pocket attention weighting
        attn_weights = self.pocket_attention(sequence_output)  # [batch, seq, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted_pooled = (sequence_output * attn_weights).sum(dim=1)

        # Combine mean pooling and attention pooling
        pooled_output = pooled_output + weighted_pooled

        return pooled_output, sequence_output


class CrossAttention(nn.Module):
    """Cross-attention module for protein-molecule interaction"""

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention: query attends to key_value

        Args:
            query: [batch, seq_len_q, dim]
            key_value: [batch, seq_len_kv, dim]

        Returns:
            Updated query representations
        """
        # Cross-attention
        attn_out, _ = self.attention(query, key_value, key_value)
        query = self.norm1(query + attn_out)

        # Feed-forward
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)

        return query


class ProteinMoleculeModel(nn.Module):
    """
    Protein-Small Molecule Interaction Prediction Model

    Predicts drug-target interactions and binding affinities.

    Architecture:
    1. ESM-2 protein encoder with binding pocket attention
    2. SMILES-based molecule encoder with multi-scale CNN + Transformer
    3. Bidirectional cross-attention fusion
    4. Multi-task prediction head (binding, affinity, interaction type)

    Output:
    - Binary interaction prediction (does it bind?)
    - Binding affinity score (how strongly?)
    - Binding site attention weights (where?)
    """

    def __init__(
        self,
        protein_dim: int = 512,
        molecule_dim: int = 512,
        hidden_dim: int = 512,
        esm_model: str = "esm2_t12_35M_UR50D",
        num_cross_attention_layers: int = 2,
        dropout: float = 0.1,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")

        # Protein encoder (ESM-2)
        self.protein_encoder = ProteinEncoder(
            model_name=esm_model, output_dim=protein_dim, device=self.device
        )

        # Molecule encoder (SMILES-based)
        self.molecule_encoder = MoleculeEncoder(
            output_dim=molecule_dim, hidden_dim=256, dropout=dropout
        )

        # Cross-attention layers (protein attends to molecule and vice versa)
        self.protein_to_molecule_attention = nn.ModuleList(
            [
                CrossAttention(dim=protein_dim, dropout=dropout)
                for _ in range(num_cross_attention_layers)
            ]
        )
        self.molecule_to_protein_attention = nn.ModuleList(
            [
                CrossAttention(dim=molecule_dim, dropout=dropout)
                for _ in range(num_cross_attention_layers)
            ]
        )

        # Binding site attention (for interpretability)
        self.binding_site_attention = nn.Sequential(
            nn.Linear(protein_dim, protein_dim // 4),
            nn.Tanh(),
            nn.Linear(protein_dim // 4, 1),
        )

        # Pharmacophore attention (for molecule interpretability)
        self.pharmacophore_attention = nn.Sequential(
            nn.Linear(molecule_dim, molecule_dim // 4),
            nn.Tanh(),
            nn.Linear(molecule_dim // 4, 1),
        )

        # Projection layers for product and diff to match concat dimension
        fusion_dim = protein_dim + molecule_dim
        self.product_projection = nn.Linear(protein_dim, fusion_dim)
        self.diff_projection = nn.Linear(protein_dim, fusion_dim)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, hidden_dim),  # concat, product, diff
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Prediction heads
        # Binary interaction prediction
        self.interaction_head = nn.Linear(hidden_dim, 1)

        # Binding affinity prediction (pIC50, pKd, etc.)
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # Interaction type classification (inhibitor, activator, etc.)
        self.interaction_type_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 interaction types
        )

    def forward(
        self,
        protein_sequences: List[str],
        molecule_sequences: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict protein-molecule interaction

        Args:
            protein_sequences: List of protein amino acid sequences
            molecule_sequences: Encoded SMILES sequences [batch, mol_len]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with:
            - interaction_logits: Binary binding prediction
            - affinity: Binding affinity score
            - interaction_type: Classification logits for interaction type
            - protein_attention: (optional) Binding site attention
            - molecule_attention: (optional) Pharmacophore attention
        """
        # Encode protein
        protein_pooled, protein_seq = self.protein_encoder(protein_sequences)

        # Encode molecule
        molecule_pooled, molecule_seq = self.molecule_encoder(molecule_sequences)

        # Cross-attention: let protein and molecule attend to each other
        for p2m_attn, m2p_attn in zip(
            self.protein_to_molecule_attention, self.molecule_to_protein_attention
        ):
            # Protein attends to molecule
            protein_seq = p2m_attn(protein_seq, molecule_seq)
            # Molecule attends to protein
            molecule_seq = m2p_attn(molecule_seq, protein_seq)

        # Get binding site attention weights
        binding_attn = self.binding_site_attention(protein_seq)  # [batch, seq, 1]
        binding_attn_weights = F.softmax(binding_attn, dim=1)

        # Get pharmacophore attention weights
        pharm_attn = self.pharmacophore_attention(molecule_seq)  # [batch, seq, 1]
        pharm_attn_weights = F.softmax(pharm_attn, dim=1)

        # Pool the cross-attended representations
        protein_final = (protein_seq * binding_attn_weights).sum(dim=1)
        molecule_final = (molecule_seq * pharm_attn_weights).sum(dim=1)

        # Combine representations
        concat = torch.cat([protein_final, molecule_final], dim=-1)
        product = protein_pooled * molecule_pooled  # Element-wise interaction
        diff = torch.abs(protein_pooled - molecule_pooled)

        # Project product and diff to match concat dimension (learned projections, not zero padding)
        product_projected = self.product_projection(product)
        diff_projected = self.diff_projection(diff)

        combined = torch.cat([concat, product_projected, diff_projected], dim=-1)

        # Fusion
        fused = self.fusion(combined)

        # Predictions
        output = {
            "interaction_logits": self.interaction_head(fused).squeeze(-1),
            "affinity": self.affinity_head(fused).squeeze(-1),
            "interaction_type": self.interaction_type_head(fused),
        }

        if return_attention:
            output["protein_attention"] = binding_attn_weights.squeeze(-1)
            output["molecule_attention"] = pharm_attn_weights.squeeze(-1)

        return output


class ProteinMoleculeDataset(Dataset):
    """Dataset for protein-molecule interactions. Expects pre-processed data from prepare/ scripts."""

    def __init__(
        self,
        data_file: Path,
        max_protein_len: int = 1000,
        max_smiles_len: int = 150,
    ):
        self.max_protein_len = max_protein_len
        self.max_smiles_len = max_smiles_len

        self.samples = []
        with open(data_file, "r") as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                protein_seq = parts[1]
                smiles = parts[2]
                label = float(parts[3])
                affinity = float(parts[4])
                self.samples.append((protein_seq, smiles, label, affinity))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        protein_seq, smiles, label, affinity = self.samples[idx]

        return {
            "protein_seq": protein_seq[: self.max_protein_len],
            "smiles": encode_smiles(smiles, self.max_smiles_len),
            "label": torch.tensor(label, dtype=torch.float32),
            "affinity": torch.tensor(affinity, dtype=torch.float32),
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    return {
        "protein_seq": [b["protein_seq"] for b in batch],
        "smiles": torch.stack([b["smiles"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "affinity": torch.stack([b["affinity"] for b in batch]),
    }


def train_epoch(model, dataloader, optimizer, device, use_affinity=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        labels = batch["label"].to(device)
        smiles = batch["smiles"].to(device)
        affinity = batch["affinity"].to(device)

        optimizer.zero_grad()

        output = model(batch["protein_seq"], smiles)

        # Binary cross-entropy for interaction prediction
        loss = F.binary_cross_entropy_with_logits(output["interaction_logits"], labels)

        # Add affinity loss for positive samples (affinity data is always available now)
        if use_affinity:
            mask = labels > 0.5
            if mask.sum() > 0:
                affinity_loss = F.mse_loss(output["affinity"][mask], affinity[mask])
                loss = loss + 0.1 * affinity_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(output["interaction_logits"]) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / len(dataloader), "accuracy": correct / total}


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_labels, all_probs = [], []
    all_affinities_true, all_affinities_pred = [], []
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            labels = batch["label"].to(device)
            smiles = batch["smiles"].to(device)
            affinity = batch["affinity"].to(device)

            output = model(batch["protein_seq"], smiles)

            loss = F.binary_cross_entropy_with_logits(
                output["interaction_logits"], labels
            )
            total_loss += loss.item()

            probs = torch.sigmoid(output["interaction_logits"])
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Collect affinity predictions for positive samples
            mask = labels > 0.5
            if mask.sum() > 0:
                all_affinities_true.extend(affinity[mask].cpu().numpy())
                all_affinities_pred.extend(output["affinity"][mask].cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Handle edge case where all labels are same class
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.5

    preds = (all_probs > 0.5).astype(float)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary", zero_division=0
    )

    # Calculate affinity RMSE if available
    affinity_rmse = 0.0
    if len(all_affinities_true) > 0:
        affinity_rmse = np.sqrt(
            mean_squared_error(all_affinities_true, all_affinities_pred)
        )

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "affinity_rmse": affinity_rmse,
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
        if val_metrics["affinity_rmse"] > 0:
            print(f"  Affinity RMSE: {val_metrics['affinity_rmse']:.4f}")

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
    parser = argparse.ArgumentParser(
        description="Train Protein-Molecule Interaction Model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing protein-molecule interaction data",
    )
    parser.add_argument("--checkpoint", type=str, default="p2m_model.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--esm-model",
        type=str,
        default="esm2_t12_35M_UR50D",
        help="ESM model name",
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
    print("PROTEIN-MOLECULE INTERACTION PREDICTION MODEL")
    print("=" * 70)
    print(f"\nData: {args.data_dir}")
    print(f"Device: {device}")
    print(f"ESM Model: {args.esm_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print()

    # Load data
    data_file = Path(args.data_dir) / "protein_molecule_interactions.tsv"
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("Run download_p2m.py first to download the data.")
        return

    dataset = ProteinMoleculeDataset(
        data_file=data_file,
        negative_ratio=1.0,
        seed=args.seed,
    )

    if args.max_samples and len(dataset) > args.max_samples:
        dataset.samples = dataset.samples[: args.max_samples]
        print(f"Limited to {args.max_samples} samples")

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

    model = ProteinMoleculeModel(
        protein_dim=512,
        molecule_dim=512,
        hidden_dim=512,
        esm_model=args.esm_model,
        num_cross_attention_layers=2,
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
