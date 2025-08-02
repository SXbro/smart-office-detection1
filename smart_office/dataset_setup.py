#!/usr/bin/env python3
"""
Smart Office Challenge - Dataset Setup Utility
Helps create and download datasets for office object detection
"""

import os
import yaml
import requests
import zipfile
from pathlib import Path
import json
import logging
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetSetup:
    def __init__(self):
        self.base_path = Path('dataset')
        self.class_names = ['person', 'chair', 'monitor', 'keyboard', 'laptop', 'phone']
        # COCO class IDs for our target classes
        self.coco_class_mapping = {
            'person': 0,
            'chair': 56,
            'laptop': 63,
            'keyboard': 66,
            'phone': 67,
            'monitor': 72  # Using 'tv' as closest to monitor
        }
    
    def create_directory_structure(self):
        """Create the required directory structure"""
        logger.info("Creating dataset directory structure...")
        
        for split in ['train', 'val', 'test']:
            (self.base_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.base_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created directory structure at: {self.base_path}")
        return True
    
    def create_data_yaml(self):
        """Create the data.yaml configuration file"""
        data_config = {
            'path': str(self.base_path.absolute()),
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = self.base_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml at: {yaml_path}")
        return yaml_path
    
    def download_sample_images(self):
        """Download sample office images for testing"""
        logger.info("Downloading sample images...")
        
        # Sample office images URLs (replace with actual URLs or use your own)
        sample_urls = [
            "https://via.placeholder.com/640x480.jpg?text=Office+Image+1",
            "https://via.placeholder.com/640x480.jpg?text=Office+Image+2", 
            "https://via.placeholder.com/640x480.jpg?text=Office+Image+3"
        ]
        
        train_images_path = self.base_path / 'train' / 'images'
        
        for i, url in enumerate(sample_urls):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    image_path = train_images_path / f"sample_{i+1}.jpg"
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded: {image_path}")
                    
                    # Create dummy label file
                    label_path = self.base_path / 'train' / 'labels' / f"sample_{i+1}.txt"
                    with open(label_path, 'w') as f:
                        # Dummy label (class 0, centered box)
                        f.write("0 0.5 0.5 0.3 0.3\n")
                    
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")
        
        logger.info("Sample images downloaded")
    
    def create_roboflow_instructions(self):
        """Create instructions for using Roboflow"""
        instructions = """
# 🤖 Using Roboflow for Dataset Setup

## Step 1: Create Roboflow Account
1. Visit: https://roboflow.com
2. Sign up for free account
3. Create new project

## Step 2: Find Office Objects Dataset
Search for datasets containing office objects:
- "office objects detection"
- "COCO office subset" 
- "workplace objects"
- "desk objects detection"

## Step 3: Download Dataset
1. Select YOLOv8 format
2. Choose train/val/test split (70/20/10 recommended)
3. Download ZIP file
4. Extract to 'dataset/' directory

## Step 4: Custom Dataset Creation
If creating your own dataset:

### Image Collection
- Collect 100-500 images per class
- Vary lighting conditions
- Include different angles and distances
- Use office/workplace environments

### Annotation Tools
- **Roboflow**: Web-based, easy to use
- **LabelImg**: Desktop tool for YOLO format
- **CVAT**: Advanced annotation platform

### Annotation Guidelines
- Draw tight bounding boxes
- Ensure consistent labeling
- Include partially visible objects
- Label all instances in each image

## Step 5: Data Format
Ensure YOLO format:
```
# Label file format (one per image)
class_id center_x center_y width height
0 0.5 0.5 0.3 0.2
1 0.2 0.3 0.15 0.25
```

Classes:
- 0: person
- 1: chair  
- 2: monitor
- 3: keyboard
- 4: laptop
- 5: phone

## Step 6: Quality Check
- Verify image-label pairs
- Check class distribution
- Validate coordinate ranges (0-1)
- Remove corrupted files
        """
        
        instructions_path = self.base_path / 'DATASET_SETUP.md'
        with open(instructions_path, 'w') as f:
            f.write(instructions)
        
        logger.info(f"Created setup instructions: {instructions_path}")
        return instructions_path
    
    def validate_dataset(self):
        """Validate dataset structure and files"""
        logger.info("Validating dataset structure...")
        
        issues = []
        
        # Check directory structure
        for split in ['train', 'val', 'test']:
            images_path = self.base_path / split / 'images'
            labels_path = self.base_path / split / 'labels'
            
            if not images_path.exists():
                issues.append(f"Missing: {images_path}")
            if not labels_path.exists():
                issues.append(f"Missing: {labels_path}")
        
        # Check data.yaml
        yaml_path = self.base_path / 'data.yaml'
        if not yaml_path.exists():
            issues.append(f"Missing: {yaml_path}")
        
        # Check for images and labels
        for split in ['train', 'val', 'test']:
            images_path = self.base_path / split / 'images'
            labels_path = self.base_path / split / 'labels'
            
            if images_path.exists():
                images = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
                labels = list(labels_path.glob('*.txt'))
                
                if len(images) == 0:
                    issues.append(f"No images found in {images_path}")
                elif len(labels) == 0:
                    issues.append(f"No labels found in {labels_path}")
                elif len(images) != len(labels):
                    issues.append(f"Image-label count mismatch in {split}: {len(images)} images, {len(labels)} labels")
        
        if issues:
            logger.warning("Dataset validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False
        else:
            logger.info("Dataset validation passed!")
            return True
    
    def create_sample_annotations(self):
        """Create sample annotation files for demonstration"""
        logger.info("Creating sample annotation files...")
        
        # Sample annotations for demo images
        sample_annotations = {
            'sample_1.txt': [
                "0 0.5 0.3 0.2 0.4",  # person
                "1 0.7 0.7 0.15 0.2"  # chair
            ],
            'sample_2.txt': [
                "2 0.4 0.2 0.3 0.25", # monitor
                "3 0.4 0.6 0.2 0.1",  # keyboard
                "4 0.6 0.5 0.25 0.2"  # laptop
            ],
            'sample_3.txt': [
                "5 0.8 0.2 0.1 0.15", # phone
                "0 0.3 0.5 0.2 0.6"   # person
            ]
        }
        
        labels_path = self.base_path / 'train' / 'labels'
        
        for filename, annotations in sample_annotations.items():
            file_path = labels_path / filename
            with open(file_path, 'w') as f:
                for annotation in annotations:
                    f.write(annotation + '\n')
        
        logger.info("Sample annotations created")
    
    def setup_complete_dataset(self):
        """Complete dataset setup process"""
        logger.info("Starting complete dataset setup...")
        
        # Create directory structure
        self.create_directory_structure()
        
        # Create configuration file
        self.create_data_yaml()
        
        # Create instructions
        self.create_roboflow_instructions()
        
        # Download sample images (for demo)
        self.download_sample_images()
        
        # Create sample annotations
        self.create_sample_annotations()
        
        # Validate setup
        is_valid = self.validate_dataset()
        
        if is_valid:
            logger.info("✅ Dataset setup completed successfully!")
            logger.info(f"📁 Dataset location: {self.base_path.absolute()}")
            logger.info("📋 Next steps:")
            logger.info("   1. Add your own images to train/val/test directories")
            logger.info("   2. Create corresponding label files")
            logger.info("   3. Run: python train.py")
            logger.info("   4. Or use Roboflow - see DATASET_SETUP.md")
        else:
            logger.error("❌ Dataset setup completed with issues")
            logger.info("📋 Please check the warnings above and fix issues")
        
        return is_valid
    
    def download_coco_subset(self):
        """Download COCO dataset subset with office objects"""
        logger.info("COCO subset download feature - Coming soon!")
        logger.info("For now, please use Roboflow or manual dataset creation")
        logger.info("See DATASET_SETUP.md for detailed instructions")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup dataset for Smart Office Challenge')
    parser.add_argument('--complete', action='store_true', 
                       help='Run complete dataset setup')
    parser.add_argument('--validate', action='store_true',
                       help='Validate existing dataset')
    parser.add_argument('--instructions', action='store_true',
                       help='Create setup instructions only')
    
    args = parser.parse_args()
    
    setup = DatasetSetup()
    
    if args.complete:
        setup.setup_complete_dataset()
    elif args.validate:
        setup.validate_dataset()
    elif args.instructions:
        setup.create_roboflow_instructions()
    else:
        # Default: show help and create basic structure
        print("Smart Office Challenge - Dataset Setup")
        print("=====================================")
        print()
        print("Options:")
        print("  --complete      Complete dataset setup with samples")
        print("  --validate      Validate existing dataset")  
        print("  --instructions  Create setup instructions")
        print()
        print("Creating basic directory structure...")
        setup.create_directory_structure()
        setup.create_data_yaml()
        setup.create_roboflow_instructions()
        print()
        print("✅ Basic setup complete!")
        print("📖 See dataset/DATASET_SETUP.md for detailed instructions")

if __name__ == "__main__":
    main()