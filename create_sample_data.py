"""
Sample Data Generator for Multi-Task Medical Imaging

This script creates dummy datasets for testing the training pipeline.
Run this to generate sample data before running the main notebook.

Usage:
    python create_sample_data.py
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

# Configuration
NUM_OA_SAMPLES = 200  # Number of OA samples to generate
NUM_OP_SAMPLES = 200  # Number of OP samples to generate
IMG_SIZE = 512        # Image size (width, height)
SEED = 42             # Random seed for reproducibility

np.random.seed(SEED)

def create_folder_structure():
    """Create the required folder structure"""
    print("Creating folder structure...")
    
    folders = [
        'data/OA/images',
        'data/OP/images',
        'checkpoints/teachers',
        'checkpoints/student',
        'results/logs',
        'results/plots',
        'results/predictions',
        'results/confusion_matrices'
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    print("✓ Folder structure created")


def generate_synthetic_xray(size=512):
    """
    Generate a synthetic X-ray-like image with random noise and patterns.
    
    Args:
        size: Image size (square)
    
    Returns:
        PIL Image
    """
    # Create base grayscale image with gradient
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create circular gradient (simulating bone density)
    center_x, center_y = 0.5, 0.5
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    gradient = 1 - (distance / distance.max())
    
    # Add random noise
    noise = np.random.randn(size, size) * 0.1
    
    # Combine
    img_array = (gradient + noise) * 200 + 50
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Convert to RGB
    img_rgb = np.stack([img_array] * 3, axis=-1)
    
    return Image.fromarray(img_rgb)


def generate_oa_dataset():
    """Generate OA (Osteoarthritis) dataset with KL grades"""
    print(f"\nGenerating {NUM_OA_SAMPLES} OA samples...")
    
    oa_data = []
    
    for i in range(NUM_OA_SAMPLES):
        # Generate image
        img = generate_synthetic_xray(IMG_SIZE)
        img_name = f'knee_{i:04d}.png'
        img_path = os.path.join('data/OA/images', img_name)
        img.save(img_path)
        
        # Assign KL grade with some distribution
        # More samples in middle grades (realistic distribution)
        grade_probs = [0.15, 0.25, 0.30, 0.20, 0.10]  # 0, 1, 2, 3, 4
        grade = np.random.choice([0, 1, 2, 3, 4], p=grade_probs)
        
        oa_data.append({
            'image_name': img_name,
            'KL_grade': grade
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{NUM_OA_SAMPLES} images")
    
    # Create DataFrame and save
    oa_df = pd.DataFrame(oa_data)
    oa_df.to_csv('data/OA/labels.csv', index=False)
    
    print(f"✓ OA dataset created: {len(oa_data)} samples")
    print("\nKL Grade Distribution:")
    print(oa_df['KL_grade'].value_counts().sort_index())
    
    return oa_df


def generate_op_dataset():
    """Generate OP (Osteoporosis) dataset with binary labels"""
    print(f"\nGenerating {NUM_OP_SAMPLES} OP samples...")
    
    op_data = []
    
    for i in range(NUM_OP_SAMPLES):
        # Generate image
        img = generate_synthetic_xray(IMG_SIZE)
        img_name = f'hip_{i:04d}.png'
        img_path = os.path.join('data/OP/images', img_name)
        img.save(img_path)
        
        # Assign binary label (0: Normal, 1: Osteoporotic)
        # Roughly balanced
        label = np.random.choice([0, 1], p=[0.6, 0.4])
        
        op_data.append({
            'image_name': img_name,
            'label': label
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{NUM_OP_SAMPLES} images")
    
    # Create DataFrame and save
    op_df = pd.DataFrame(op_data)
    op_df.to_csv('data/OP/labels.csv', index=False)
    
    print(f"✓ OP dataset created: {len(op_data)} samples")
    print("\nLabel Distribution:")
    print(op_df['label'].value_counts().sort_index())
    print("  0 = Normal")
    print("  1 = Osteoporotic")
    
    return op_df


def verify_dataset():
    """Verify that datasets were created correctly"""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    # Check OA dataset
    oa_csv_exists = os.path.exists('data/OA/labels.csv')
    oa_images_exist = len(os.listdir('data/OA/images')) > 0
    
    print(f"\nOA Dataset:")
    print(f"  ✓ labels.csv exists: {oa_csv_exists}")
    print(f"  ✓ Images folder populated: {oa_images_exist}")
    
    if oa_csv_exists:
        oa_df = pd.read_csv('data/OA/labels.csv')
        print(f"  ✓ Total samples: {len(oa_df)}")
        print(f"  ✓ Columns: {list(oa_df.columns)}")
    
    # Check OP dataset
    op_csv_exists = os.path.exists('data/OP/labels.csv')
    op_images_exist = len(os.listdir('data/OP/images')) > 0
    
    print(f"\nOP Dataset:")
    print(f"  ✓ labels.csv exists: {op_csv_exists}")
    print(f"  ✓ Images folder populated: {op_images_exist}")
    
    if op_csv_exists:
        op_df = pd.read_csv('data/OP/labels.csv')
        print(f"  ✓ Total samples: {len(op_df)}")
        print(f"  ✓ Columns: {list(op_df.columns)}")
    
    # Check folder structure
    print(f"\nFolder Structure:")
    required_folders = [
        'data/OA/images',
        'data/OP/images',
        'checkpoints/teachers',
        'checkpoints/student',
        'results/plots'
    ]
    
    all_exist = True
    for folder in required_folders:
        exists = os.path.exists(folder)
        print(f"  {'✓' if exists else '✗'} {folder}")
        if not exists:
            all_exist = False
    
    print("\n" + "="*60)
    if all_exist and oa_csv_exists and op_csv_exists:
        print("✅ SETUP COMPLETE - Ready to train!")
        print("="*60)
        print("\nNext steps:")
        print("1. Open multi_task_medical_imaging.ipynb")
        print("2. Run all cells in order")
        print("3. Monitor training progress")
    else:
        print("⚠️  SETUP INCOMPLETE - Please fix the issues above")
        print("="*60)


def main():
    """Main function to generate sample datasets"""
    print("="*60)
    print("SAMPLE DATA GENERATOR")
    print("Multi-Task Medical Imaging Training Pipeline")
    print("="*60)
    
    print("\nThis script will create:")
    print(f"  • {NUM_OA_SAMPLES} synthetic OA X-ray images")
    print(f"  • {NUM_OP_SAMPLES} synthetic OP X-ray images")
    print(f"  • Label CSV files for both datasets")
    print(f"  • Required folder structure")
    
    print("\nNOTE: These are synthetic images for testing only!")
    print("Replace with real medical images for actual training.\n")
    
    # Confirm
    response = input("Continue? (yes/no): ").lower().strip()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Create folder structure
    create_folder_structure()
    
    # Generate datasets
    oa_df = generate_oa_dataset()
    op_df = generate_op_dataset()
    
    # Verify
    verify_dataset()
    
    print("\n✓ Sample data generation complete!")
    print("\nYou can now run the training notebook:")
    print("  jupyter notebook multi_task_medical_imaging.ipynb")


if __name__ == "__main__":
    main()
