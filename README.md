# T5-G2T: Sign Language Translation with Contrastive Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.20+-orange.svg)](https://huggingface.co/transformers/)

A approach for Sign Language Translation that combines **T5 transformer architecture** with **contrastive learning (CLIP)** to translate gloss sequences into natural language text. Employs a novel two-stage training approach: contrastive pre-training for cross-modal representation learning, followed by fine-tuning for direct gloss-to-text translation.


## Requirements

### System Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for GPU training)
- **Memory**: 16GB+ RAM recommended
- **Storage**: 10GB+ free space

### Dependencies
```bash
torch>=1.9.0
torchvision>=0.10.0
torchaudio>=0.9.0
transformers>=4.20.0
datasets>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
nltk>=3.7
spacy>=3.4.0
opencv-python>=4.5.0
pillow>=8.3.0
tensorboard>=2.8.0
loguru>=0.6.0
torchinfo>=1.7.0
torchviz>=0.3.0
seaborn>=0.11.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
scipy>=1.7.0
tqdm>=4.62.0
pyyaml>=6.0
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/tinh2044/T5-G2T
cd T5-G2T

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Data Preparation

```bash
# Download Phoenix-2014-T dataset
# Place the dataset in ./data/Phoenxi-2014-T/
# Expected structure:
# ./data/Phoenxi-2014-T/
# ├── train.csv
# ├── dev.csv
# ├── test.csv
# └── all.csv
```

### 3. Training

#### Stage 1: CLIP Pre-training
```bash
# Train CLIP model for cross-modal representation learning
python train_clip.py \
    --config configs/phoenix-2014-t.yaml \
    --batch-size 4 \
    --epochs 80 \
    --lr 1e-3 \
    --output_dir ./outputs/clip/phoenix-2014-T
```

#### Stage 2: G2T Fine-tuning
```bash
# Fine-tune for gloss-to-text translation
python train_g2t.py \
    --config configs/phoenix-2014-t.yaml \
    --batch-size 4 \
    --epochs 80 \
    --lr 1e-4 \
    --finetune ./outputs/clip/phoenix-2014-T/best_checkpoint.pth \
    --output_dir ./outputs/g2t/phoenix-2014-T
```

#### Using Training Scripts
```bash
# Automated training with predefined configurations
bash scripts/train/train_clip-Phoenix-2014-T.sh
bash scripts/train/train_g2t-Phoenix-2014-T.sh
```

### 4. Evaluation

```bash
# Evaluate trained model
python train_g2t.py \
    --config configs/phoenix-2014-t.yaml \
    --eval \
    --finetune ./outputs/g2t/phoenix-2014-T/best_checkpoint.pth \
    --output_dir ./outputs/evaluation
```

### 5. Inference

```python
import torch
from transformers import T5Tokenizer
from models import GlossTextCLIP

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
model = GlossTextCLIP.from_pretrained("./outputs/g2t/phoenix-2014-T/best_checkpoint.pth")
model.eval()

# Prepare input
gloss_sequence = "JETZT WETTER MORGEN DONNERSTAG"
inputs = tokenizer(gloss_sequence, return_tensors="pt")

# Generate translation
with torch.no_grad():
    outputs = model.generate(
        inputs,
        num_beams=5,
        max_length=128,
        early_stopping=True
    )

# Decode result
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Gloss: {gloss_sequence}")
print(f"Translation: {translation}")
```

## Configuration

### Model Configuration (`configs/phoenix-2014-t.yaml`)
```yaml
name: T5-S2T
data:
  path: ./data/Phoenxi-2014-T
training:
  wandb: disabled
  scale_embedding: False
model:
  tokenizer: google-t5/t5-base
  transformer: google-t5/t5-base
  sign_proj: True
```

### Training Parameters
| Parameter | CLIP | G2T | Description |
|-----------|------|-----|-------------|
| Learning Rate | 1e-3 | 1e-4 | Initial learning rate |
| Batch Size | 4 | 4 | Training batch size |
| Epochs | 80 | 80 | Training epochs |
| Scheduler | Cosine | Cosine | LR scheduler type |
| Optimizer | AdamW | AdamW | Optimization algorithm |
| Weight Decay | 0.0 | 0.0 | L2 regularization |

## Performance

### Expected Results on Phoenix-2014-T
| Metric | Score | Description |
|--------|-------|-------------|
| BLEU-4 | 15-25% | 4-gram BLEU score |
| ROUGE-L | 35-45% | Longest common subsequence |
| ROUGE-1 | 40-50% | Unigram overlap |
| ROUGE-2 | 25-35% | Bigram overlap |

### Training Time
- **CLIP Stage**: ~8-12 hours (4x V100 GPUs)
- **G2T Stage**: ~6-10 hours (4x V100 GPUs)
- **Total**: ~14-22 hours

## Testing

Run unit tests to verify model functionality:

```bash
# Run all tests
python -m pytest test_model.py -v

# Run specific test
python test_model.py TestGlossTextCLIP.test_forward_clip_output_shape
```

## Monitoring

### TensorBoard
```bash
# Launch TensorBoard
tensorboard --logdir ./outputs

# View training metrics at http://localhost:6006
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


<div align="center">

**Made with ❤️ for the accessibility community**

[⭐ Star this repo](https://github.com/tinh2044/T5-G2T) | [🍴 Fork it](https://github.com/tinh2044/T5-G2T/fork) | [📝 Report Issues](https://github.com/tinh2044/T5-G2T/issues)

</div>
