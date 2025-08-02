#!/usr/bin/env python3
"""
Smart Office Challenge - Complete Pipeline Runner
Orchestrates the entire training, evaluation, and deployment pipeline
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
import logging
from config import Config

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self):
        self.config = Config
        self.start_time = time.time()
        
    def run_command(self, command, description="Running command"):
        """Execute a command and handle errors"""
        logger.info(f"{description}: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                logger.info(f"Output: {result.stdout.strip()}")
            
            return True, result.stdout
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            return False, e.stderr
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'ultralytics',
            'torch', 
            'torchvision',
            'opencv-python',
            'streamlit',
            'pandas',
            'matplotlib',
            'plotly'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Install missing packages with: pip install -r requirements.txt")
            return False
        
        logger.info("✅ All dependencies are installed")
        return True
    
    def setup_dataset(self, create_sample=False):
        """Setup dataset structure"""
        logger.info("Setting up dataset...")
        
        if create_sample:
            success, output = self.run_command(
                [sys.executable, "dataset_setup.py", "--complete"],
                "Creating dataset with samples"
            )
        else:
            success, output = self.run_command(
                [sys.executable, "dataset_setup.py"],
                "Creating basic dataset structure"
            )
        
        if not success:
            logger.error("Dataset setup failed")
            return False
        
        logger.info("✅ Dataset setup completed")
        return True
    
    def train_model(self, model_size='n', epochs=50, img_size=640):
        """Train the model"""
        logger.info(f"Starting model training (YOLOv8{model_size}, {epochs} epochs)...")
        
        # Check if dataset exists
        if not (self.config.DATASET_PATH / "data.yaml").exists():
            logger.warning("Dataset not found, setting up basic structure...")
            if not self.setup_dataset():
                return False
        
        command = [
            sys.executable, "train.py",
            "--model", model_size,
            "--epochs", str(epochs),
            "--img-size", str(img_size)
        ]
        
        success, output = self.run_command(command, "Training model")
        
        if not success:
            logger.error("Model training failed")
            return False
        
        # Check if model was created
        if os.path.exists(self.config.MODEL_SAVE_PATH):
            logger.info(f"✅ Model training completed: {self.config.MODEL_SAVE_PATH}")
            return True
        else:
            logger.error("Model file not found after training")
            return False
    
    def evaluate_model(self, model_path=None):
        """Evaluate the trained model"""
        if model_path is None:
            model_path = self.config.MODEL_SAVE_PATH
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return False
        
        logger.info(f"Evaluating model: {model_path}")
        
        command = [
            sys.executable, "evaluate.py",
            "--model", model_path
        ]
        
        success, output = self.run_command(command, "Evaluating model")
        
        if not success:
            logger.error("Model evaluation failed")
            return False
        
        logger.info("✅ Model evaluation completed")
        return True
    
    def launch_app(self, port=8501):
        """Launch the Streamlit application"""
        logger.info(f"Launching Streamlit app on port {port}...")
        
        command = [
            "streamlit", "run", "app.py",
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        logger.info("Starting Streamlit server...")
        logger.info(f"App will be available at: http://localhost:{port}")
        
        try:
            # Run streamlit (this will block)
            subprocess.run(command)
        except KeyboardInterrupt:
            logger.info("Streamlit app stopped by user")
        except Exception as e:
            logger.error(f"Failed to start Streamlit app: {e}")
            return False
        
        return True
    
    def run_complete_pipeline(self, model_size='n', epochs=50, skip_training=False):
        """Run the complete pipeline"""
        logger.info("🚀 Starting complete Smart Office Challenge pipeline...")
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            return False
        
        # Step 2: Setup dataset (if needed)
        if not (self.config.DATASET_PATH / "data.yaml").exists():
            logger.info("Dataset not found, creating sample dataset...")
            if not self.setup_dataset(create_sample=True):
                return False
        
        # Step 3: Train model (if not skipping)
        if not skip_training:
            if not self.train_model(model_size=model_size, epochs=epochs):
                return False
        else:
            logger.info("Skipping training step")
            if not os.path.exists(self.config.MODEL_SAVE_PATH):
                logger.error(f"Model not found: {self.config.MODEL_SAVE_PATH}")
                logger.info("Either train a model or set --skip-training=false")
                return False
        
        # Step 4: Evaluate model
        if not self.evaluate_model():
            return False
        
        # Step 5: Show summary
        self.print_pipeline_summary()
        
        logger.info("✅ Complete pipeline finished successfully!")
        logger.info("🌐 Launch the web app with: streamlit run app.py")
        
        return True
    
    def print_pipeline_summary(self):
        """Print pipeline execution summary"""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("🏢 SMART OFFICE CHALLENGE - PIPELINE SUMMARY")
        print("="*60)
        print(f"⏱️  Total execution time: {elapsed_time:.2f} seconds")
        print(f"📁 Dataset location: {self.config.DATASET_PATH}")
        print(f"🤖 Model location: {self.config.MODEL_SAVE_PATH}")
        print(f"📊 Results location: {self.config.RESULTS_DIR}")
        print()
        print("📋 Next Steps:")
        print("   1. Launch web app: streamlit run app.py")
        print("   2. Upload office images for detection")
        print("   3. Review evaluation results in evaluation_results/")
        print("   4. Fine-tune model if needed")
        print()
        print("🎯 Hackathon Submission Ready!")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Smart Office Challenge Pipeline Runner')
    
    # Pipeline options
    parser.add_argument('--complete', action='store_true',
                       help='Run complete pipeline (setup, train, evaluate)')
    parser.add_argument('--train-only', action='store_true',
                       help='Run training only')
    parser.add_argument('--eval-only', action='store_true', 
                       help='Run evaluation only')
    parser.add_argument('--app-only', action='store_true',
                       help='Launch web app only')
    parser.add_argument('--setup-only', action='store_true',
                       help='Setup dataset only')
    
    # Training parameters
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    
    # Other options
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step in complete pipeline')
    parser.add_argument('--port', type=int, default=8501,
                       help='Port for Streamlit app')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies only')
    
    args = parser.parse_args()
    
    runner = PipelineRunner()
    
    try:
        if args.check_deps:
            runner.check_dependencies()
        elif args.setup_only:
            runner.setup_dataset(create_sample=True)
        elif args.train_only:
            runner.train_model(
                model_size=args.model_size,
                epochs=args.epochs,
                img_size=args.img_size
            )
        elif args.eval_only:
            runner.evaluate_model()
        elif args.app_only:
            runner.launch_app(port=args.port)
        elif args.complete:
            runner.run_complete_pipeline(
                model_size=args.model_size,
                epochs=args.epochs,
                skip_training=args.skip_training
            )
        else:
            # Default: show help and run basic setup
            print("Smart Office Challenge - Pipeline Runner")
            print("======================================")
            print()
            print("Quick start options:")
            print("  --complete       Run full pipeline")
            print("  --setup-only     Setup dataset")
            print("  --train-only     Train model")
            print("  --eval-only      Evaluate model")
            print("  --app-only       Launch web app")
            print("  --check-deps     Check dependencies")
            print()
            print("Running basic dependency check...")
            runner.check_dependencies()
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()