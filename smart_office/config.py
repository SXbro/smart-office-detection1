#!/usr/bin/env python3
"""
Smart Office Challenge - Configuration Settings
Centralized configuration for all scripts
"""

from pathlib import Path
import os

class Config:
    """Configuration class for Smart Office Challenge"""
    
    # Project information
    PROJECT_NAME = "Smart Office Challenge"
    VERSION = "1.0.0"
    
    # Dataset configuration
    DATASET_PATH = Path("dataset")
    DATA_YAML = DATASET_PATH / "data.yaml"
    
    # Class configuration
    CLASS_NAMES = ['person', 'chair', 'monitor', 'keyboard', 'laptop', 'phone']
    NUM_CLASSES = len(CLASS_NAMES)
    
    # Class colors for visualization
    CLASS_COLORS = {
        'person': '#FF6B6B',      # Red
        'chair': '#4ECDC4',       # Teal
        'monitor': '#45B7D1',     # Blue
        'keyboard': '#96CEB4',    # Green
        'laptop': '#FFEAA7',      # Yellow
        'phone': '#DDA0DD'        # Purple
    }
    
    # COCO class mapping (for COCO dataset usage)
    COCO_CLASS_MAPPING = {
        'person': 0,
        'chair': 56,
        'laptop': 63,
        'keyboard': 66,
        'phone': 67,
        'monitor': 72  # Using 'tv' as closest to monitor
    }
    
    # Model configuration
    DEFAULT_MODEL_SIZE = 'n'  # YOLOv8 nano (fastest)
    AVAILABLE_MODEL_SIZES = ['n', 's', 'm', 'l', 'x']
    MODEL_SAVE_PATH = "best_office_model.pt"
    
    # Training configuration
    DEFAULT_EPOCHS = 50
    DEFAULT_IMG_SIZE = 640
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_LEARNING_RATE = 0.01
    
    # Training parameters
    TRAIN_PARAMS = {
        'batch': DEFAULT_BATCH_SIZE,
        'cache': False,
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
        'lr0': DEFAULT_LEARNING_RATE,
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
    
    # Evaluation configuration
    EVAL_CONF_THRESHOLD = 0.25
    EVAL_IOU_THRESHOLD = 0.45
    EVAL_MAX_DET = 300
    
    # Evaluation parameters
    EVAL_PARAMS = {
        'batch': 16,
        'conf': EVAL_CONF_THRESHOLD,
        'iou': EVAL_IOU_THRESHOLD,
        'max_det': EVAL_MAX_DET,
        'half': False,
        'device': 'cpu',
        'dnn': False,
        'plots': True,
        'save_json': True,
        'save_hybrid': False,
        'verbose': True,
        'split': 'test',
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'visualize': False,
        'augment': False,
        'agnostic_nms': False,
        'retina_masks': False
    }
    
    # Directory paths
    RUNS_DIR = Path("runs")
    RESULTS_DIR = Path("evaluation_results")
    LOGS_DIR = Path("logs")
    
    # Web app configuration
    STREAMLIT_CONFIG = {
        'page_title': PROJECT_NAME,
        'page_icon': '🏢',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }
    
    # Default confidence thresholds for web app
    DEFAULT_CONF_THRESHOLD = 0.25
    DEFAULT_IOU_THRESHOLD = 0.45
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File extensions
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    MODEL_EXTENSION = '.pt'
    
    # Dataset split ratios (if creating custom dataset)
    DATASET_SPLIT = {
        'train': 0.7,
        'val': 0.2,
        'test': 0.1
    }
    
    # Hardware configuration
    USE_GPU = True  # Will fallback to CPU if CUDA not available
    
    @classmethod
    def get_device(cls):
        """Get the appropriate device (GPU/CPU) - Python 3.10 compatible"""
        try:
            import torch
            if cls.USE_GPU and torch.cuda.is_available():
                return 'cuda'
        except ImportError:
            pass
        return 'cpu'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATASET_PATH,
            cls.RUNS_DIR,
            cls.RESULTS_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        # Check class configuration
        if len(cls.CLASS_NAMES) != cls.NUM_CLASSES:
            errors.append("CLASS_NAMES length doesn't match NUM_CLASSES")
        
        # Check model size
        if cls.DEFAULT_MODEL_SIZE not in cls.AVAILABLE_MODEL_SIZES:
            errors.append(f"Invalid DEFAULT_MODEL_SIZE: {cls.DEFAULT_MODEL_SIZE}")
        
        # Check dataset split ratios
        total_split = sum(cls.DATASET_SPLIT.values())
        if abs(total_split - 1.0) > 0.001:
            errors.append(f"Dataset split ratios don't sum to 1.0: {total_split}")
        
        # Check thresholds
        if not (0.0 <= cls.DEFAULT_CONF_THRESHOLD <= 1.0):
            errors.append("DEFAULT_CONF_THRESHOLD must be between 0.0 and 1.0")
        
        if not (0.0 <= cls.DEFAULT_IOU_THRESHOLD <= 1.0):
            errors.append("DEFAULT_IOU_THRESHOLD must be between 0.0 and 1.0")
        
        return errors
    
    @classmethod
    def print_config_summary(cls):
        """Print configuration summary"""
        print("=" * 50)
        print(f"{cls.PROJECT_NAME} - Configuration Summary")
        print("=" * 50)
        print(f"Version: {cls.VERSION}")
        print(f"Classes: {cls.NUM_CLASSES} ({', '.join(cls.CLASS_NAMES)})")
        print(f"Default Model: YOLOv8{cls.DEFAULT_MODEL_SIZE}")
        print(f"Device: {cls.get_device()}")
        print(f"Dataset Path: {cls.DATASET_PATH}")
        print(f"Image Size: {cls.DEFAULT_IMG_SIZE}")
        print(f"Batch Size: {cls.DEFAULT_BATCH_SIZE}")
        print(f"Epochs: {cls.DEFAULT_EPOCHS}")
        print("=" * 50)

# Environment-specific overrides
def load_environment_config():
    """Load configuration from environment variables"""
    # Override with environment variables if set
    if os.getenv('OFFICE_MODEL_SIZE'):
        Config.DEFAULT_MODEL_SIZE = os.getenv('OFFICE_MODEL_SIZE')
    
    if os.getenv('OFFICE_EPOCHS'):
        Config.DEFAULT_EPOCHS = int(os.getenv('OFFICE_EPOCHS'))
    
    if os.getenv('OFFICE_BATCH_SIZE'):
        Config.DEFAULT_BATCH_SIZE = int(os.getenv('OFFICE_BATCH_SIZE'))
    
    if os.getenv('OFFICE_IMG_SIZE'):
        Config.DEFAULT_IMG_SIZE = int(os.getenv('OFFICE_IMG_SIZE'))
    
    if os.getenv('OFFICE_DATASET_PATH'):
        Config.DATASET_PATH = Path(os.getenv('OFFICE_DATASET_PATH'))
        Config.DATA_YAML = Config.DATASET_PATH / "data.yaml"

# Load environment config on import
load_environment_config()

# Validate configuration
config_errors = Config.validate_config()
if config_errors:
    print("Configuration Errors:")
    for error in config_errors:
        print(f"  - {error}")
    raise ValueError("Invalid configuration settings")

if __name__ == "__main__":
    # Print configuration when run directly
    Config.print_config_summary()
    
    # Create directories
    Config.create_directories()
    print("✅ Directories created successfully")