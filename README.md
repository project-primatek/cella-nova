# Cella Nova

**Cella Nova** is a comprehensive cell modeling platform that uses deep learning to predict biomolecular interactions. The platform combines protein language models, graph neural networks, and attention mechanisms to model the complex interaction networks within cells.

## Features

### Protein-Protein Interaction (PPI) Prediction
Predict whether two proteins interact using:
- **ESM-2 protein language model** for sequence embeddings
- **Graph Neural Networks** for protein interaction network topology
- **Structure encoding** via AlphaFold contact maps
- **Siamese network architecture** for pairwise prediction

### Protein-DNA Interaction Prediction
Predict transcription factor binding and protein-DNA interactions using:
- **ESM-2** for protein sequence encoding
- **CNN + Bi-LSTM + Attention** for DNA sequence encoding
- **Cross-attention mechanism** for modeling protein-DNA interactions
- **Binding affinity prediction** alongside binary classification

## Data Sources

### Protein-Protein Interactions
- **STRING Database**: Known and predicted protein-protein interactions with confidence scores
- **AlphaFold Database**: Predicted 3D structures for structural feature extraction

### Protein-DNA Interactions
- **JASPAR**: Transcription factor binding motifs
- **UniProt**: DNA-binding protein sequences and annotations
- **ENCODE**: ChIP-seq experimental binding data

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/cella-nova.git
cd cella-nova

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Download Data

```bash
# Download proteome and AlphaFold structures
python download_proteome.py --species "Homo sapiens" --threads 20

# Download STRING interactions
python download_string.py --taxon-id 9606 --score 700

# Download protein-DNA interaction data
python download_pdna.py --species human
```

### 2. Train Models

```bash
# Train protein-protein interaction model
python model_p2p.py --data-dir data/homo_sapiens --epochs 50

# Train protein-DNA interaction model
python model_p2d.py --data-dir data/protein_dna --epochs 50
```

### 3. Make Predictions

```python
from model_p2p import PPIModel
from model_p2d import ProteinDNAModel

# Load trained models
ppi_model = PPIModel.load("ppi_model.pt")
p2d_model = ProteinDNAModel.load("pdna_model.pt")

# Predict protein-protein interaction
score = ppi_model.predict(protein_a_seq, protein_b_seq)

# Predict protein-DNA binding
binding_prob, affinity = p2d_model.predict(protein_seq, dna_seq)
```

## Model Performance

### PPI Model
| Metric | Score |
|--------|-------|
| AUC | 0.9999 |
| Precision | 0.99 |
| Recall | 1.00 |
| F1 Score | 0.995 |

### Protein-DNA Model (3 epochs)
| Metric | Score |
|--------|-------|
| AUC | 0.7087 |
| F1 Score | 0.5745 |
| Precision | 0.5870 |
| Recall | 0.5625 |

## Project Structure

```
cella-nova/
├── data/
│   └── {species}/
│       ├── proteins.fasta
│       ├── sequences/
│       ├── structures/
│       └── string/
├── download_proteome.py    # Download protein sequences and structures
├── download_string.py      # Download STRING interaction data
├── download_pdna.py        # Download protein-DNA interaction data
├── model_p2p.py            # Protein-protein interaction model
├── model_p2d.py            # Protein-DNA interaction model
├── requirements.txt
└── README.md
```

## Architecture Overview

### PPI Model
```
Protein A Sequence ──► ESM-2 Encoder ──┐
                                       ├──► Cross-Modal Fusion ──► MLP ──► Interaction Score
Protein B Sequence ──► ESM-2 Encoder ──┤
                                       │
Network Topology ────► GNN Encoder ────┘
```

### Protein-DNA Model
```
Protein Sequence ──► ESM-2 Encoder ──────────────┐
                                                  ├──► Cross-Attention ──► MLP ──► Binding Prediction
DNA Sequence ──► CNN ──► Bi-LSTM ──► Attention ──┘
```

## References

### Databases
- [STRING Database](https://string-db.org/) - Protein-protein interaction networks
- [AlphaFold Database](https://alphafold.ebi.ac.uk/) - Protein structure predictions
- [JASPAR](https://jaspar.genereg.net/) - Transcription factor binding profiles
- [UniProt](https://www.uniprot.org/) - Protein sequence and annotation
- [ENCODE](https://www.encodeproject.org/) - Encyclopedia of DNA Elements

### Key Papers
- Jumper et al. (2021) - "Highly accurate protein structure prediction with AlphaFold" - *Nature*
- Lin et al. (2023) - "Evolutionary-scale prediction of atomic-level protein structure with a language model" - *Science*
- Szklarczyk et al. (2023) - "The STRING database in 2023" - *Nucleic Acids Research*

## License

MIT