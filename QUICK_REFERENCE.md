# Quick Reference Guide

## 🚀 Quick Start Commands

### 1. Setup Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install torch torchvision torchaudio
pip install jupyter numpy pandas matplotlib seaborn scikit-learn pillow opencv-python tqdm
```

### 2. Launch Jupyter Notebook
```bash
jupyter notebook multi_task_medical_imaging.ipynb
```

### 3. Verify Setup
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"MPS Available: {torch.backends.mps.is_available()}")
```

---

## 📁 Sample Dataset Creation

### Create Dummy Dataset for Testing
```python
import os
import pandas as pd
import numpy as np
from PIL import Image

# Create folder structure
os.makedirs('data/OA/images', exist_ok=True)
os.makedirs('data/OP/images', exist_ok=True)

# Generate sample OA images and labels
oa_data = []
for i in range(100):  # 100 sample images
    # Create random grayscale image (512x512)
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img_path = f'data/OA/images/knee_{i:03d}.png'
    Image.fromarray(img).save(img_path)
    
    # Random KL grade (0-4)
    grade = np.random.randint(0, 5)
    oa_data.append({'image_name': f'knee_{i:03d}.png', 'KL_grade': grade})

# Save OA labels
oa_df = pd.DataFrame(oa_data)
oa_df.to_csv('data/OA/labels.csv', index=False)
print(f"Created {len(oa_data)} OA samples")
print(oa_df['KL_grade'].value_counts())

# Generate sample OP images and labels
op_data = []
for i in range(100):  # 100 sample images
    # Create random grayscale image
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img_path = f'data/OP/images/hip_{i:03d}.png'
    Image.fromarray(img).save(img_path)
    
    # Random binary label
    label = np.random.randint(0, 2)
    op_data.append({'image_name': f'hip_{i:03d}.png', 'label': label})

# Save OP labels
op_df = pd.DataFrame(op_data)
op_df.to_csv('data/OP/labels.csv', index=False)
print(f"Created {len(op_data)} OP samples")
print(op_df['label'].value_counts())

print("\n✓ Sample dataset created!")
print("Run the notebook to test the pipeline.")
```

---

## 🔧 Common Adjustments

### Reduce Memory Usage
```python
CONFIG = {
    'batch_size': 4,        # From 8
    'img_size': 384,        # From 512
    'num_epochs': 30,       # From 50
}
```

### Use Smaller Model
```python
# In model definition, replace:
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
```

### Adjust Learning Rate
```python
# If training is unstable
CONFIG['learning_rate'] = 5e-5  # Reduce

# If training is too slow
CONFIG['learning_rate'] = 2e-4  # Increase
```

---

## 📊 Monitoring Training

### Check Progress
```python
# After training
import matplotlib.pyplot as plt

# Plot losses
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss_oa'], label='OA Train')
plt.plot(history['val_loss_oa'], label='OA Val')
plt.legend()
plt.title('OA Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_loss_op'], label='OP Train')
plt.plot(history['val_loss_op'], label='OP Val')
plt.legend()
plt.title('OP Loss')
plt.show()
```

### Check Metrics
```python
# Load best checkpoint
checkpoint = torch.load('checkpoints/student/student_best.pth')
print(f"Best Epoch: {checkpoint['epoch']}")
print(f"OA QWK: {checkpoint['val_metrics']['oa_qwk']:.4f}")
print(f"OP AUC: {checkpoint['val_metrics']['op_auc']:.4f}")
```

---

## 🐛 Debugging Tips

### Data Loading Issues
```python
# Test data loading
from torch.utils.data import DataLoader

test_loader = DataLoader(oa_train_dataset_full, batch_size=2, shuffle=False)
images, labels, names = next(iter(test_loader))

print(f"Batch shape: {images.shape}")
print(f"Labels: {labels}")
print(f"Image names: {names}")

# Visualize a sample
import matplotlib.pyplot as plt
img = images[0].permute(1, 2, 0).numpy()
img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Denormalize
plt.imshow(np.clip(img, 0, 1))
plt.title(f"Label: {labels[0]}")
plt.show()
```

### Model Output Check
```python
# Test model forward pass
model = StudentMultiTaskModel()
model = model.to(device)
model.eval()

dummy_input = torch.randn(1, 3, 512, 512).to(device)
oa_out, op_out = model(dummy_input)

print(f"OA output shape: {oa_out.shape}")  # Should be [1, 5]
print(f"OP output shape: {op_out.shape}")  # Should be [1, 1]
```

### Loss Calculation Check
```python
# Test loss functions
criterion_oa = WeightedCrossEntropyLoss()
criterion_op = FocalLoss()

# Dummy predictions and labels
oa_preds = torch.randn(4, 5)  # Batch=4, Classes=5
oa_labels = torch.tensor([0, 2, 3, 4])

op_preds = torch.randn(4, 1)  # Batch=4, Binary
op_labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

loss_oa = criterion_oa(oa_preds, oa_labels)
loss_op = criterion_op(op_preds, op_labels)

print(f"OA Loss: {loss_oa.item()}")
print(f"OP Loss: {loss_op.item()}")
```

---

## 💡 Performance Tips

### Speed Up Training
```python
# 1. Use DataLoader with persistent workers (if not Mac)
# num_workers=2, persistent_workers=True

# 2. Enable pin_memory
# DataLoader(..., pin_memory=True)

# 3. Use mixed precision (if supported on MPS)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

### Reduce Overfitting
```python
# Increase dropout
self.oa_head = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),  # Increase from 0.3
    ...
)

# Add weight decay
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4  # Increase from 1e-5
)

# Increase data augmentation
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # Increase
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Increase
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 📈 Expected Timeline (Mac M4, 16GB RAM)

| Phase | Task | Time (approx) |
|-------|------|---------------|
| 1 | Environment setup | 10-15 min |
| 2 | Data preparation | 5-10 min |
| 3 | Teacher OA training (30 epochs) | 30-45 min |
| 4 | Teacher OP training (30 epochs) | 30-45 min |
| 5 | Student training (50 epochs) | 2-3 hours |
| 6 | Evaluation & visualization | 5-10 min |
| **Total** | | **~4-5 hours** |

*Times vary based on dataset size*

---

## 🔄 Training Pipeline Summary

```
1. Data Preparation
   ↓
2. Train OA Teacher (30 epochs)
   ↓
3. Train OP Teacher (30 epochs)
   ↓
4. Freeze Teachers
   ↓
5. Train Student with KD (50 epochs)
   - Forward pass on both tasks
   - Get teacher soft targets
   - Compute task losses + KD losses
   - Apply PCGrad
   - Update student
   ↓
6. Evaluate & Visualize
   ↓
7. Save best model & results
```

---

## 📋 Checklist Before Training

**Data:**
- [ ] Images in correct folders
- [ ] CSV files created
- [ ] Image names match CSV
- [ ] Labels are correct (0-4 for OA, 0-1 for OP)
- [ ] Sufficient samples (>100 per task minimum)

**Environment:**
- [ ] PyTorch installed
- [ ] MPS available (for Mac M4)
- [ ] All dependencies installed
- [ ] Jupyter notebook launches

**Configuration:**
- [ ] Batch size appropriate for RAM
- [ ] Learning rate set
- [ ] Loss weights balanced
- [ ] Paths in CONFIG correct

**Optional:**
- [ ] Test on dummy data first
- [ ] Monitor system resources
- [ ] Set up logging
- [ ] Backup important data

---

## 🎯 Success Criteria

### Minimum Viable Model
- OA Accuracy: >60%
- OP Accuracy: >80%
- Training completes without errors
- Checkpoints save correctly

### Good Model
- OA QWK: >0.70
- OP AUC: >0.85
- Stable training curves
- Good generalization (train vs val)

### Excellent Model
- OA QWK: >0.85
- OP AUC: >0.95
- Minimal overfitting
- Fast convergence

---

## 🔐 Important Reminders

1. **Medical data is sensitive** - Follow privacy regulations
2. **Not for clinical use** - Research only
3. **Validate results** - Cross-check with domain experts
4. **Document everything** - Keep logs for reproducibility
5. **Backup regularly** - Save checkpoints frequently

---

## 📚 Additional Resources

### PyTorch Documentation
- Models: https://pytorch.org/vision/stable/models.html
- Data Loading: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- MPS Backend: https://pytorch.org/docs/stable/notes/mps.html

### Medical Imaging
- KL Grading: https://en.wikipedia.org/wiki/Kellgren%E2%80%93Lawrence_grade
- DICOM Processing: https://pydicom.github.io/
- Medical Image Preprocessing: https://simpleitk.org/

### Research Papers
- Original methodology from your document
- EfficientNet paper
- CBAM paper
- PCGrad paper

---

**Happy Training! 🚀**
