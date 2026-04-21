# Multi-Task Medical Imaging: Setup Guide

## 📋 Overview

This project implements a complete multi-task deep learning framework for medical imaging:
- **Task 1 (OA)**: Osteoarthritis KL grading (5-class: 0-4)
- **Task 2 (OP)**: Osteoporosis detection (binary)

### Key Features
✅ Domain-specific teacher models  
✅ Knowledge distillation (soft targets)  
✅ PCGrad for gradient conflict resolution  
✅ CBAM attention mechanism  
✅ Curriculum learning (easy → moderate → hard)  
✅ Optimized for Mac M4 (MPS backend)  

---

## 📁 Required Folder Structure

Create this exact folder structure in your project directory:

```
project_root/
│
├── multi_task_medical_imaging.ipynb    # Main training notebook
│
├── data/
│   ├── OA/                             # Osteoarthritis dataset
│   │   ├── images/                     # X-ray knee images
│   │   │   ├── knee_001.png
│   │   │   ├── knee_002.png
│   │   │   └── ...
│   │   │
│   │   └── labels.csv                  # OA labels file
│   │
│   └── OP/                             # Osteoporosis dataset
│       ├── images/                     # X-ray hip/spine images
│       │   ├── hip_001.png
│       │   ├── hip_002.png
│       │   └── ...
│       │
│       └── labels.csv                  # OP labels file
│
├── checkpoints/                        # (Auto-created) Model checkpoints
│   ├── teachers/                       # Teacher model weights
│   └── student/                        # Student model weights
│
└── results/                            # (Auto-created) Training outputs
    ├── logs/                           # Training logs
    ├── plots/                          # Training curves
    ├── predictions/                    # Prediction visualizations
    └── confusion_matrices/             # Confusion matrix plots
```

---

## 📝 CSV Label File Format

### OA Dataset (`data/OA/labels.csv`)

```csv
image_name,KL_grade
knee_001.png,0
knee_002.png,1
knee_003.png,2
knee_004.png,3
knee_005.png,4
knee_006.png,0
...
```

**Columns:**
- `image_name`: Filename of the X-ray image (must match files in `data/OA/images/`)
- `KL_grade`: Kellgren-Lawrence grade (0, 1, 2, 3, or 4)

### OP Dataset (`data/OP/labels.csv`)

```csv
image_name,label
hip_001.png,0
hip_002.png,1
hip_003.png,0
hip_004.png,1
...
```

**Columns:**
- `image_name`: Filename of the X-ray image (must match files in `data/OP/images/`)
- `label`: Binary label (0 = Normal, 1 = Osteoporotic)

---

## 🔧 Installation & Setup

### 1. Create Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate     # On Windows

# Upgrade pip
pip install --upgrade pip
```

### 2. Install PyTorch (Mac M4 with MPS)

```bash
# For Mac M4 (Apple Silicon)
pip install torch torchvision torchaudio
```

### 3. Install Other Dependencies

```bash
pip install jupyter notebook
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install pillow opencv-python
pip install tqdm
```

### 4. Verify MPS Availability

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"PyTorch version: {torch.__version__}")
```

---

## 🚀 Usage Instructions

### Step 1: Prepare Your Data

1. **Organize images** into the folder structure above
2. **Create CSV files** with labels in the correct format
3. **Verify paths** match between CSV and actual image locations

### Step 2: Configure Training

Open the notebook and adjust the `CONFIG` dictionary:

```python
CONFIG = {
    'batch_size': 8,        # Reduce to 4 if out of memory
    'num_epochs': 50,       # Adjust based on dataset size
    'learning_rate': 1e-4,
    'temperature': 3.0,     # KD temperature
    
    # Loss weights (should sum to ~1.0)
    'alpha': 0.3,           # OA task loss weight
    'beta': 0.3,            # OP task loss weight
    'lambda_1': 0.2,        # OA distillation loss
    'lambda_2': 0.2,        # OP distillation loss
}
```

### Step 3: Run Training

Execute the notebook cells in order:

1. **Phase 0**: Setup and imports
2. **Phase 1**: Data preparation
3. **Phase 2**: Model architecture
4. **Phase 3**: Loss functions
5. **Phase 4**: PCGrad implementation
6. **Phase 5**: Training functions
7. **Phase 7**: Main training pipeline
   - Train OA teacher
   - Train OP teacher
   - Train student with KD
8. **Phase 8**: Evaluation and visualization

### Step 4: Monitor Training

Watch for these indicators:
- **Training curves**: Should show decreasing loss
- **Validation metrics**: Monitor QWK (OA) and AUC (OP)
- **GPU/MPS memory**: Reduce batch size if OOM
- **Gradient conflicts**: PCGrad should resolve conflicts

---

## 💾 Saved Outputs

### Checkpoints
- `checkpoints/teachers/teacher_oa_best.pth` - Best OA teacher
- `checkpoints/teachers/teacher_op_best.pth` - Best OP teacher
- `checkpoints/student/student_best.pth` - Best student model
- `checkpoints/student/student_epoch_*.pth` - Periodic checkpoints

### Results
- `results/plots/training_history.png` - Training curves
- `results/confusion_matrices/cm_oa.png` - OA confusion matrix
- `results/confusion_matrices/cm_op.png` - OP confusion matrix
- `results/predictions/predictions_OA.png` - OA sample predictions
- `results/predictions/predictions_OP.png` - OP sample predictions
- `results/final_results.json` - Complete metrics summary

---

## 📊 Expected Performance

Based on the methodology, expect:

### OA Task
- **Accuracy**: 70-85%
- **QWK**: 0.75-0.90 (target)
- **Challenge**: Grade 2 classification (borderline cases)

### OP Task
- **Accuracy**: 85-95%
- **AUC**: 0.90-0.98 (target)
- **F1-Score**: 0.85-0.95

### Training Time (Mac M4, 16GB)
- Teacher models: ~30-60 min each
- Student model: ~2-3 hours
- Total: ~4-5 hours (depending on dataset size)

---

## 🔍 Troubleshooting

### Out of Memory (OOM)

**Symptoms**: Process crashes, MPS errors

**Solutions**:
```python
CONFIG['batch_size'] = 4  # Reduce from 8
# Or enable gradient accumulation
```

### Slow Training

**Solutions**:
- Reduce image size: `CONFIG['img_size'] = 384` (from 512)
- Use EfficientNet-B0 instead of B3
- Enable mixed precision (if supported)

### Poor Convergence

**Check**:
1. Learning rate (try 5e-5 or 2e-4)
2. Loss weights (ensure balanced)
3. Data quality (check labels)
4. Class imbalance (add class weights)

### Data Loading Issues

**Verify**:
```python
# Check data paths
import os
print(os.path.exists('data/OA/images/'))
print(os.path.exists('data/OA/labels.csv'))

# Check CSV format
import pandas as pd
df = pd.read_csv('data/OA/labels.csv')
print(df.head())
print(df['KL_grade'].value_counts())
```

---

## 🧪 Hyperparameter Tuning

### Critical Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `learning_rate` | 1e-4 | 5e-5 to 2e-4 | Convergence speed |
| `temperature` | 3.0 | 2.0 to 5.0 | KD softness |
| `alpha` (OA weight) | 0.3 | 0.2 to 0.4 | Task balance |
| `beta` (OP weight) | 0.3 | 0.2 to 0.4 | Task balance |
| `lambda_1` (OA KD) | 0.2 | 0.1 to 0.3 | Distillation strength |
| `batch_size` | 8 | 4 to 16 | Memory/stability |

### Tuning Strategy
1. **Start with defaults**
2. **Monitor validation metrics** (QWK, AUC)
3. **Adjust one parameter at a time**
4. **Keep training curves smooth**

---

## 📈 Evaluation Metrics

### OA Task (Multi-class)
- **Accuracy**: Overall correctness
- **QWK** (Quadratic Weighted Kappa): Ordinal classification quality
- **Macro F1**: Balance across all classes
- **Confusion Matrix**: Per-class performance

### OP Task (Binary)
- **Accuracy**: Overall correctness
- **AUC-ROC**: Discrimination ability
- **F1-Score**: Precision-recall balance
- **Sensitivity/Specificity**: Clinical relevance

---

## 🔬 For Research Use

### Reproducibility
```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```

### Ablation Studies
Test different components:
1. **Without KD**: Set `lambda_1 = lambda_2 = 0`
2. **Without PCGrad**: Use standard optimizer
3. **Without CBAM**: Remove attention module
4. **Without curriculum**: Use full dataset from start

### Logging for Papers
```python
# Add to training loop
import logging
logging.basicConfig(filename='training.log', level=logging.INFO)
```

### Generate Attention Maps
```python
# For interpretability
from torch.nn import functional as F
# Use hooks to extract CBAM attention weights
```

---

## 📚 References

1. **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network"
2. **PCGrad**: Yu et al., "Gradient Surgery for Multi-Task Learning"
3. **CBAM**: Woo et al., "CBAM: Convolutional Block Attention Module"
4. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling"

---

## ⚠️ Important Notes

### Data Privacy
- Medical images are sensitive
- Ensure proper IRB approval
- Anonymize patient data
- Follow HIPAA/GDPR guidelines

### Model Limitations
- Not FDA approved
- For research only
- Requires clinical validation
- Should not replace radiologist diagnosis

### Citation
If you use this code, please cite appropriately in your research.

---

## 🆘 Support

### Common Issues
1. **ImportError**: Install missing packages
2. **CUDA not found**: Use MPS for Mac M4
3. **Permission errors**: Check file paths
4. **Memory errors**: Reduce batch size

### Getting Help
- Check error messages carefully
- Verify data format
- Test on small dataset first
- Monitor GPU/MPS usage

---

## ✅ Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] PyTorch with MPS support installed
- [ ] All dependencies installed
- [ ] Folder structure created
- [ ] OA images in `data/OA/images/`
- [ ] OP images in `data/OP/images/`
- [ ] `data/OA/labels.csv` created (correct format)
- [ ] `data/OP/labels.csv` created (correct format)
- [ ] CSV files match image filenames
- [ ] CONFIG parameters reviewed
- [ ] MPS device detected

**Ready to train!** 🚀

---

## 📞 Contact & Support

For issues specific to this implementation, check:
1. PyTorch documentation
2. scikit-learn documentation
3. Ensure data format is correct
4. Verify all paths

**Good luck with your research!** 🎓
