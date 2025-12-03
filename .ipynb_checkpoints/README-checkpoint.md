# Active Missing Modality Completion (AMC) for Multimodal Learning

This project implements an active missing modality completion framework for multimodal learning, designed to handle incomplete modality data in various datasets like ENRICO, MIMIC-IV, and ADNI.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [Configuration](#configuration)
- [Debugging](#debugging)
- [Datasets](#datasets)
- [Models](#models)
- [Training Scripts](#training-scripts)

## Project Overview

This repository contains implementations of various multimodal transformer architectures designed for:
- Handling missing modalities in multimodal learning
- Cross-modal attention mechanisms
- Token-level fusion strategies
- Difference attention mechanisms (DiffAttn)

Key features:
- Support for multiple modalities (images, wireframes, etc.)
- Various fusion strategies (weighted fusion, token fusion, etc.)
- Sparse MoE (Mixture of Experts) support
- Configurable network architectures via YAML files

## Directory Structure

```
amc/
├── acm_utilize/          # Utility functions (YAML parsing, etc.)
├── config/               # Configuration files for different models
├── dataset/              # Dataset implementations
├── model/                # Model definitions (transformers, attention, etc.)
├── running_scripts/      # Shell scripts for running experiments
├── scattermoe/           # Sparse MoE implementations
├── train/                # Training scripts
├── environment.yml       # Conda environment specification
└── environment_install.md # Installation instructions
```

## Dependencies

This project requires:
- Python 3.10 or 3.11
- PyTorch 2.3.0 or higher
- CUDA 12.1 support (for GPU acceleration)
- Additional packages listed in `environment.yml`

### Required Python Libraries

- torch, torchvision, torchaudio
- einops
- opencv-python
- tqdm
- yaml
- numpy
- pandas
- scikit-learn
- timm
- easydict

## Setup

### 1. Environment Setup

To set up the environment, you can either use the provided conda environment or install dependencies manually:

**Option A: Using conda environment**

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate imbalance_modality
```

**Option B: Manual installation**

```bash
# Create a new environment
conda create -n active_missing python=3.10 -y
conda activate active_missing

# Install PyTorch with CUDA support
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install additional dependencies
pip install einops opencv-python tqdm
```

### 2. Set Environment Variable

The project uses an environment variable `ACTIVE_ROOT` to locate the root directory:

```bash
export ACTIVE_ROOT="/path/to/your/amc"
```

For persistent setup, add this to your `.bashrc` or `.zshrc` file.

## Running the Project

### Training

The project supports training on multiple datasets. The main training script is located at `train/train.py`.

**Basic usage:**

```bash
# Set the environment variable
export ACTIVE_ROOT="/path/to/your/amc"

# Run training with default configuration
python train/train.py --model_config config/multi_modal_config.yml

# Example with specific parameters
python train/train.py \
  --epochs 100 \
  --batch_size 32 \
  --lr 5e-4 \
  --path model_ckpt \
  --cpt_name enrico_img \
  --enrico_path data/enrico \
  --model_config config/multi_modal_config.yml
```

### Using Running Scripts

The repository includes shell scripts for different experiments in `running_scripts/`:

```bash
# ENRICO dataset training
bash running_scripts/enrico.sh

# MIMIC dataset training
bash running_scripts/mimic.sh

# ADNI dataset training
bash running_scripts/adni.sh
```

### GPU Training

To specify GPU device:

```bash
CUDA_VISIBLE_DEVICES=0 python train/train.py --model_config config/multi_modal_config.yml
```

## Configuration

### YAML Configuration Files

Model configurations are stored in the `config/` directory as YAML files. Example configuration:

```yaml
modality:
  screenImg:
    max_freq: 1
    freq_bands: 6
    freq_base: 2
    additional_dim: 384
    modality_latent_len: 16
    modelity_weight: 0.9
  screenWireframeImg:
    max_freq: 1
    freq_bands: 6
    freq_base: 2
    additional_dim: 384
    modality_latent_len: 16
    modelity_weight: 0.1

network_type: MultimodalTransformerWF  # or MultimodalTransformer, MultimodalTransformerTokenF, etc.

network:
  hidden_size: 64
  latent_length: 32
  n_heads: 8
  n_blocks: 4
  dropout_rate: 0.1
  pred_dim: 20
  mlp_ratio: 4
```

### Command Line Arguments

The main training script supports these arguments:

- `--seed`: Random seed (default: 123)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--num_workers`: Number of data loading workers (default: 4)
- `--lr`: Learning rate (default: 5e-4)
- `--path`: Model checkpoint path (default: "model_ckpt")
- `--cpt_name`: Checkpoint name (default: "enrico_img")
- `--enrico_path`: Path to ENRICO dataset
- `--train`: Whether to train or not (default: True)
- `--model_config`: Path to model configuration YAML file

## Debugging

### Common Issues and Solutions

1. **CUDA Error: Out of memory**
   - Reduce batch size: `--batch_size 16` or lower
   - Use fewer workers: `--num_workers 2` or lower
   - Adjust model size in configuration

2. **Missing dataset files**
   - Ensure dataset paths are correct
   - Verify dataset structure matches expected format
   - Check file permissions

3. **Configuration errors**
   - Verify YAML syntax
   - Ensure all required fields are present
   - Check that modality names match between configuration and dataset

### Debugging Tips

1. **Enable detailed logging:**
   ```bash
   # Run with increased verbosity by modifying the training script
   # Add print statements in forward functions to track tensor shapes
   ```

2. **Test data loading:**
   ```python
   # Test dataset loading independently
   from dataset.enrico_dataset import get_dataset
   train_ds, val_ds, test_ds = get_dataset('path/to/enrico')
   print(f"Train dataset size: {len(train_ds)}")
   sample = train_ds[0]
   print(f"Sample keys: {list(sample.keys())}")
   ```

3. **Validate model initialization:**
   ```python
   # Check if model parameters are properly initialized
   from model.dense_model import MultimodalTransformerWF
   from acm_utilize.yml_parser import YmlConfig
   
   config = YmlConfig('config/multi_modal_config.yml')
   # Initialize model and check parameter count
   ```

4. **Check for NaN/Inf values:**
   - Monitor loss values during training
   - Add gradient clipping and check for exploding gradients
   - Enable tensorboard logging for detailed monitoring

### Debugging Commands

```bash
# Run a quick test with fewer epochs and smaller batch size
python train/train.py --epochs 1 --batch_size 2 --model_config config/multi_modal_config.yml

# Debug dataset loading
python -c "from dataset.enrico_dataset import get_dataset; train, val, test = get_dataset('data/enrico'); print(len(train), len(val), len(test)); print(train[0])"

# Check model instantiation
python -c "from model.dense_model import MultimodalTransformerWF; from acm_utilize.yml_parser import YmlConfig; import torch; config = YmlConfig('config/multi_modal_config.yml'); device = torch.device('cpu'); modality_config = {}; [modality_config.update({kk: config.parse_to_modality(config.obj.modality[kk])}) for kk in config.obj.modality]; model = MultimodalTransformerWF(device, modality_config, **config.obj.network); print(f'Parameters: {sum(p.numel() for p in model.parameters())}')"
```

## Datasets

The project supports multiple datasets:

1. **ENRICO**: UI design dataset with screenshots and wireframes
2. **MIMIC-IV**: Medical imaging dataset
3. **ADNI**: Alzheimer's Disease Neuroimaging Initiative dataset

### Dataset Structure

ENRICO dataset should be structured as:

```
enrico/
├── design_topics.csv
├── screenshots/
│   ├── <screen_id>.jpg
├── wireframes/
│   ├── <screen_id>.png
└── hierarchies/
    ├── <screen_id>.json
```

## Models

The project includes various transformer architectures:

- **SinglemodalTransformer**: Transformer for single modality
- **MultimodalTransformer**: Basic multimodal transformer
- **MultimodalTransformerWF**: Weighted fusion multimodal transformer
- **MultimodalTransformerTokenF**: Token fusion multimodal transformer
- **DiffAttn**: Difference attention mechanism
- **SparseMoeBlock**: Sparse Mixture of Experts implementation
- **DiffSparseTransformerBlock**: Difference attention with sparse MoE

## Training Scripts

Available training scripts:

- `train/train.py`: Main training script for ENRICO dataset
- `train/train_mimic.py`: Training script for MIMIC dataset
- `train/train_adni.py`: Training script for ADNI dataset

## Contributing

When contributing to this project:

1. Follow the existing code style and structure
2. Update documentation as needed
3. Test changes thoroughly
4. Add appropriate configuration files for new features

## License

The MIMIC-IV and ADNI datasets require licenses. Please obtain the license and then contact us for processed datasets.

ENRICO dataset: https://github.com/pliang279/MultiBench

TCGA dataset support coming soon.