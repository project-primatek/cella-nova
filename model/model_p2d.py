#!/usr/bin/env python3
"""
Protein-DNA Interaction Prediction Model
=========================================

Multi-modal model for predicting protein-DNA interactions:
1. ESM-2 for protein sequence encoding
2. CNN + Attention for DNA sequence encoding
3. Cross-attention fusion between protein and DNA representations

Usage:
    python model_pdna.py --data-dir data/pdna --epochs 50
    python model_pdna.py --data-dir data/pdna --epochs 10 --max-samples 5000
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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

# DNA nucleotide encoding
DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
DNA_VOCAB_SIZE = 5
DNA_PAD_IDX = 4


def encode_dna(sequence: str, max_length: int = 200) -> torch.Tensor:
    """Encode DNA sequence to tensor"""
    sequence = sequence.upper().replace("U", "T")
    indices = [DNA_VOCAB.get(nt, DNA_PAD_IDX) for nt in sequence[:max_length]]

    # Pad to max_length
    if len(indices) < max_length:
        indices.extend([DNA_PAD_IDX] * (max_length - len(indices)))

    return torch.tensor(indices, dtype=torch.long)


class DNAEncoder(nn.Module):
    """
    DNA sequence encoder using CNN + Bi-LSTM + Self-Attention

    Captures local motifs (CNN), sequential dependencies (Bi-LSTM),
    and long-range interactions (Self-Attention) - MPS compatible
    """

    def __init__(
        self,
        vocab_size: int = DNA_VOCAB_SIZE,
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

        # Nucleotide embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=DNA_PAD_IDX)

        # CNN for local motif detection
        self.conv_layers = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Bi-LSTM for sequential dependencies (MPS compatible)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Multi-head self-attention (MPS compatible - no mask needed)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

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
        Encode DNA sequence

        Args:
            x: DNA sequence indices [batch_size, seq_len]

        Returns:
            Tuple of (pooled_output, sequence_output)
            - pooled_output: [batch_size, output_dim]
            - sequence_output: [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len = x.shape

        # Create padding mask
        padding_mask = x == DNA_PAD_IDX  # [batch, seq_len]

        # Embed nucleotides
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # CNN expects [batch, channels, seq_len]
        conv_input = embedded.transpose(1, 2)
        conv_out = self.conv_layers(conv_input)  # [batch, hidden_dim, seq_len]
        conv_out = conv_out.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        # Bi-LSTM encoding
        lstm_out, _ = self.lstm(conv_out)  # [batch, seq_len, hidden_dim]

        # Self-attention (without key_padding_mask for MPS compatibility)
        # Instead, we zero out padded positions after attention
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(lstm_out + attn_out)  # Residual connection

        # Zero out padded positions
        mask = (~padding_mask).float().unsqueeze(-1)  # [batch, seq_len, 1]
        attn_out = attn_out * mask

        # Project to output dimension
        sequence_output = self.projection(attn_out)  # [batch, seq_len, output_dim]

        # Pooled output (mean over non-padded positions)
        pooled_output = (sequence_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(
            min=1
        )

        return pooled_output, sequence_output


class ProteinEncoder(nn.Module):
    """ESM-2 based protein encoder"""

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

        return pooled_output, sequence_output


class CrossAttention(nn.Module):
    """Cross-attention module for protein-DNA interaction"""

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


class ProteinDNAModel(nn.Module):
    """
    Protein-DNA Interaction Prediction Model

    Combines:
    1. ESM-2 protein encoder
    2. CNN + Transformer DNA encoder
    3. Cross-attention fusion
    4. Interaction prediction head
    """

    def __init__(
        self,
        protein_dim: int = 512,
        dna_dim: int = 512,
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

        # DNA encoder
        self.dna_encoder = DNAEncoder(
            output_dim=dna_dim, hidden_dim=256, dropout=dropout
        )

        # Cross-attention layers (protein attends to DNA and vice versa)
        self.protein_to_dna_attention = nn.ModuleList(
            [
                CrossAttention(dim=protein_dim, dropout=dropout)
                for _ in range(num_cross_attention_layers)
            ]
        )
        self.dna_to_protein_attention = nn.ModuleList(
            [
                CrossAttention(dim=dna_dim, dropout=dropout)
                for _ in range(num_cross_attention_layers)
            ]
        )

        # Projection layers for product and diff to match concat dimension
        fusion_dim = protein_dim + dna_dim
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
        self.interaction_head = nn.Linear(hidden_dim, 1)
        self.binding_affinity_head = nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def forward(
        self,
        protein_sequences: List[str],
        dna_sequences: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict protein-DNA interaction

        Args:
            protein_sequences: List of protein amino acid sequences
            dna_sequences: Encoded DNA sequences [batch, dna_len]

        Returns:
            Tuple of (interaction_logits, binding_affinity)
        """
        # Encode protein
        protein_pooled, protein_seq = self.protein_encoder(protein_sequences)

        # Encode DNA
        dna_pooled, dna_seq = self.dna_encoder(dna_sequences)

        # Cross-attention: let protein and DNA attend to each other
        for p2d_attn, d2p_attn in zip(
            self.protein_to_dna_attention, self.dna_to_protein_attention
        ):
            # Protein attends to DNA
            protein_seq = p2d_attn(protein_seq, dna_seq)
            # DNA attends to protein
            dna_seq = d2p_attn(dna_seq, protein_seq)

        # Pool the cross-attended representations
        protein_final = protein_seq.mean(dim=1)
        dna_final = dna_seq.mean(dim=1)

        # Combine representations
        concat = torch.cat([protein_final, dna_final], dim=-1)
        product = protein_pooled * dna_pooled  # Element-wise interaction
        diff = torch.abs(protein_pooled - dna_pooled)

        # Project product and diff to match concat dimension (learned projections, not zero padding)
        product_projected = self.product_projection(product)
        diff_projected = self.diff_projection(diff)

        combined = torch.cat([concat, product_projected, diff_projected], dim=-1)

        # Fusion
        fused = self.fusion(combined)

        # Predictions
        interaction_logits = self.interaction_head(fused).squeeze(-1)
        binding_affinity = self.binding_affinity_head(fused).squeeze(-1)

        return interaction_logits, binding_affinity


class ProteinDNADataset(Dataset):
    """Dataset for protein-DNA interactions. Expects pre-processed data from prepare/ scripts."""

    def __init__(
        self,
        data_file: Path,
        max_protein_len: int = 1000,
        max_dna_len: int = 200,
    ):
        self.max_protein_len = max_protein_len
        self.max_dna_len = max_dna_len

        self.samples = []
        with open(data_file, "r") as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                protein_seq = parts[1]
                dna_seq = parts[2]
                label = float(parts[3])
                self.samples.append((protein_seq, dna_seq, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        protein_seq, dna_seq, label = self.samples[idx]

        return {
            "protein_seq": protein_seq[: self.max_protein_len],
            "dna_seq": encode_dna(dna_seq, self.max_dna_len),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    return {
        "protein_seq": [b["protein_seq"] for b in batch],
        "dna_seq": torch.stack([b["dna_seq"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        labels = batch["label"].to(device)
        dna_seq = batch["dna_seq"].to(device)

        optimizer.zero_grad()

        logits, _ = model(batch["protein_seq"], dna_seq)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
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
            dna_seq = batch["dna_seq"].to(device)

            logits, _ = model(batch["protein_seq"], dna_seq)

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
    parser = argparse.ArgumentParser(description="Train Protein-DNA Interaction Model")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing protein-DNA interaction data",
    )
    parser.add_argument("--checkpoint", type=str, default="pdna_model.pt")
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
    print("PROTEIN-DNA INTERACTION PREDICTION MODEL")
    print("=" * 70)
    print(f"\nData: {args.data_dir}")
    print(f"Device: {device}")
    print(f"ESM Model: {args.esm_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print()

    # Load data
    data_file = Path(args.data_dir) / "protein_dna_interactions.tsv"
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("Run download_pdna.py first to download the data.")
        return

    dataset = ProteinDNADataset(
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

    model = ProteinDNAModel(
        protein_dim=512,
        dna_dim=512,
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
