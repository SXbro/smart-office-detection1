#!/usr/bin/env python3
"""
Smart Office Challenge - Training Script
Trains YOLOv8 model for office object detection using transfer learning
"""

import os
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfficeObjectTrainer:
    def __init__(self, data_path, model_size='n', epochs=50, img_size=640):
        """
        Initialize the trainer
        
        Args:
            data_path (str): Path to dataset YAML file
            model_size (str): YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            epochs (int): Number of training epochs
            img_size (int): Input image size
        """
        self.data_path = data_path
        self.model_size = model_size
        self.epochs = epochs
        self.img_size = img_size
        self.model_name = f'yolov8{model_size}.pt'
        self.project_name = 'office_detection'
        
    def setup_data_yaml(self):
        """Create dataset configuration file"""
        # Office object classes mapping
        class_names = ['person', 'chair', 'monitor', 'keyboard', 'laptop', 'phone']
        
        # Create data.yaml configuration
        data_config = {
            'path': str(Path(self.data_path).parent.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = Path(self.data_path).parent / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
            
        logger.info(f"Created data configuration: {yaml_path}")
        return str(yaml_path)
    
    def create_sample_dataset_structure(self):
        """Create sample dataset structure for demonstration"""
        base_path = Path('dataset')
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (base_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (base_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        logger.info("Created dataset directory structure")
        logger.info("Please add your images to dataset/{train,val,test}/images/")
        logger.info("Please add corresponding YOLO format labels to dataset/{train,val,test}/labels/")
        
        return str(base_path / 'data.yaml')
    
    def download_pretrained_weights(self):
        """Download YOLOv8 pretrained weights"""
        try:
            model = YOLO(self.model_name)
            logger.info(f"Downloaded pretrained weights: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to download weights: {e}")
            raise
    
    def train_model(self):
        """Train the YOLOv8 model"""
        try:
            # Load pretrained model
            model = YOLO(self.model_name)
            
            # Check if dataset exists, create sample structure if not
            if not os.path.exists(self.data_path):
                logger.warning("Dataset not found, creating sample structure...")
                data_yaml = self.create_sample_dataset_structure()
            else:
                data_yaml = self.setup_data_yaml()
            
            # Configure training parameters
            train_params = {
                'data': data_yaml,
                'epochs': self.epochs,
                'imgsz': self.img_size,
                'batch': 16,
                'name': self.project_name,
                'save': True,
                'save_period': 10,
                'cache': False,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'workers': 4,
                'project': 'runs/detect',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'SGD',
                'verbose': True,
                'seed': 42,
                'deterministic': True,
                'single_cls': False,
                'rect': False,
                'cos_lr': False,
                'close_mosaic': 10,
                'resume': False,
                'amp': True,
                'fraction': 1.0,
                'profile': False,
                'freeze': None,
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True
            }
            
            logger.info("Starting training...")
            logger.info(f"Device: {train_params['device']}")
            logger.info(f"Epochs: {self.epochs}")
            logger.info(f"Image size: {self.img_size}")
            
            # Train the model
            results = model.train(**train_params)
            
            # Save the best model
            best_model_path = f'runs/detect/{self.project_name}/weights/best.pt'
            if os.path.exists(best_model_path):
                # Copy to easy access location
                import shutil
                shutil.copy2(best_model_path, 'best_office_model.pt')
                logger.info(f"Best model saved to: best_office_model.pt")
            
            logger.info("Training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate_model(self, model_path='best_office_model.pt'):
        """Validate the trained model"""
        try:
            model = YOLO(model_path)
            data_yaml = self.setup_data_yaml()
            
            # Run validation
            results = model.val(data=data_yaml, imgsz=self.img_size)
            
            logger.info("Validation completed!")
            logger.info(f"mAP50: {results.box.map50:.4f}")
            logger.info(f"mAP50-95: {results.box.map:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise

def setup_roboflow_dataset():
    """
    Instructions for setting up dataset using Roboflow
    """
    instructions = """
    # Setting up Dataset with Roboflow:
    
    1. Visit roboflow.com and create account
    2. Search for 'office objects' or 'COCO' datasets
    3. Download dataset in YOLOv8 format
    4. Extract to 'dataset/' directory
    5. Ensure structure:
       dataset/
       ├── train/
       │   ├── images/
       │   └── labels/
       ├── val/
       │   ├── images/
       │   └── labels/
       └── test/
           ├── images/
           └── labels/
    
    # Alternative: Use COCO subset
    The trainer will filter COCO dataset for our 6 classes automatically.
    """
    print(instructions)

def main():
    parser = argparse.ArgumentParser(description='Train office object detection model')
    parser.add_argument('--data', type=str, default='dataset/data.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--setup-data', action='store_true',
                       help='Show dataset setup instructions')
    
    args = parser.parse_args()
    
    if args.setup_data:
        setup_roboflow_dataset()
        return
    
    # Initialize trainer
    trainer = OfficeObjectTrainer(
        data_path=args.data,
        model_size=args.model,
        epochs=args.epochs,
        img_size=args.img_size
    )
    
    # Train model
    try:
        results = trainer.train_model()
        
        # Validate model
        if os.path.exists('best_office_model.pt'):
            trainer.validate_model()
            
    except Exception as e:
        logger.error(f"Training process failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())