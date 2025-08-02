#!/usr/bin/env python3
"""
Smart Office Challenge - Python 3.10 Compatibility Checker
Verifies system compatibility and provides setup guidance
"""

import sys
import subprocess
import importlib
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    print(f"   Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 10:
        print("   ✅ Python 3.10 detected - Perfect!")
        return True
    elif version.major == 3 and version.minor >= 8:
        print(f"   ⚠️  Python {version.major}.{version.minor} detected - Should work but 3.10 recommended")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} detected - May have compatibility issues")
        print("   📝 Recommendation: Use Python 3.10 for best compatibility")
        return False

def check_system_info():
    """Display system information"""
    print("\n💻 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Processor: {platform.processor()}")

def check_required_packages():
    """Check if required packages can be imported"""
    print("\n📦 Checking package compatibility...")
    
    # Core packages that must work
    core_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'matplotlib': 'matplotlib',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'yaml': 'PyYAML'
    }
    
    # Optional but important packages
    optional_packages = {
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn'
    }
    
    def test_import(package_name, pip_name):
        try:
            importlib.import_module(package_name)
            print(f"   ✅ {pip_name}")
            return True
        except ImportError:
            print(f"   ❌ {pip_name} - Not installed")
            return False
    
    print("\n   Core packages:")
    core_success = all(test_import(pkg, pip) for pkg, pip in core_packages.items())
    
    print("\n   ML/AI packages:")
    ml_success = all(test_import(pkg, pip) for pkg, pip in optional_packages.items())
    
    if not core_success:
        print("\n   ⚠️  Some core packages are missing. Install with:")
        print("      pip install numpy pandas matplotlib Pillow opencv-python PyYAML")
    
    if not ml_success:
        print("\n   ⚠️  Some ML packages are missing. Install all with:")
        print("      pip install -r requirements.txt")
    
    return core_success and ml_success

def check_gpu_availability():
    """Check GPU availability for training"""
    print("\n🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ GPU available: {gpu_name}")
            print(f"   📊 GPU count: {gpu_count}")
            print(f"   💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠️  No GPU available - will use CPU")
            print("   💡 Training will be slower but still functional")
    except ImportError:
        print("   ⚠️  PyTorch not installed - cannot check GPU")

def check_disk_space():
    """Check available disk space"""
    print("\n💾 Checking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        print(f"   📊 Free space: {free_gb:.1f} GB")
        
        if free_gb < 2:
            print("   ⚠️  Low disk space - may need more for datasets and models")
        elif free_gb < 5:
            print("   ✅ Adequate space for basic usage")
        else:
            print("   ✅ Plenty of space available")
            
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")

def create_compatible_requirements():
    """Create Python 3.10 compatible requirements file"""
    print("\n📝 Creating Python 3.10 compatible requirements...")
    
    compatible_requirements = """# Smart Office Challenge - Python 3.10 Compatible Requirements

# Core ML and Computer Vision (3.10 compatible versions)
ultralytics==8.0.196
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.0.76
Pillow==10.0.0
numpy==1.24.3

# Data Processing and Analysis
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Web Application
streamlit==1.25.0
plotly==5.15.0

# Utilities
PyYAML==6.0.1
tqdm==4.65.0
psutil==5.9.5

# Optional: For enhanced functionality
imageio==2.31.1

# Development tools (optional)
pytest==7.4.0
black==23.7.0
"""
    
    with open("requirements_py310.txt", "w") as f:
        f.write(compatible_requirements)
    
    print("   ✅ Created requirements_py310.txt with compatible versions")
    print("   💡 Install with: pip install -r requirements_py310.txt")

def provide_setup_instructions():
    """Provide setup instructions for Python 3.10"""
    print("\n🚀 Python 3.10 Setup Instructions:")
    print("="*50)
    
    print("\n1️⃣  Virtual Environment Setup (Recommended):")
    print("   python -m venv office_env")
    
    if platform.system() == "Windows":
        print("   office_env\\Scripts\\activate")
    else:
        print("   source office_env/bin/activate")
    
    print("\n2️⃣  Install Dependencies:")
    print("   pip install --upgrade pip")
    print("   pip install -r requirements_py310.txt")
    print("   # OR use the original: pip install -r requirements.txt")
    
    print("\n3️⃣  Verify Installation:")
    print("   python check_compatibility.py")
    
    print("\n4️⃣  Quick Start:")
    print("   python run_pipeline.py --check-deps")
    print("   python run_pipeline.py --complete")
    
    print("\n5️⃣  Launch Web App:")
    print("   streamlit run app.py")
    
    print("\n💡 Troubleshooting Tips:")
    print("   - If torch installation fails, try: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("   - For GPU support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("   - If streamlit fails: pip install streamlit --upgrade")

def run_compatibility_test():
    """Run complete compatibility test"""
    print("🏢 Smart Office Challenge - Python 3.10 Compatibility Check")
    print("="*60)
    
    # Check Python version
    python_ok = check_python_version()
    
    # System info
    check_system_info()
    
    # Package check
    packages_ok = check_required_packages()
    
    # GPU check
    check_gpu_availability()
    
    # Disk space
    check_disk_space()
    
    # Create compatible requirements
    create_compatible_requirements()
    
    # Overall assessment
    print("\n🎯 Compatibility Assessment:")
    print("="*30)
    
    if python_ok and packages_ok:
        print("✅ System is ready for Smart Office Challenge!")
        print("🚀 You can proceed with: python run_pipeline.py --complete")
    elif python_ok:
        print("⚠️  Python version is compatible, but packages need installation")
        print("📦 Run: pip install -r requirements_py310.txt")
    else:
        print("❌ Compatibility issues detected")
        print("📝 Please follow the setup instructions below")
    
    # Setup instructions
    provide_setup_instructions()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check Python 3.10 compatibility')
    parser.add_argument('--create-requirements', action='store_true',
                       help='Create Python 3.10 compatible requirements file')
    parser.add_argument('--quick', action='store_true',
                       help='Quick check only')
    
    args = parser.parse_args()
    
    if args.create_requirements:
        create_compatible_requirements()
    elif args.quick:
        check_python_version()
        check_required_packages()
    else:
        run_compatibility_test()

if __name__ == "__main__":
    main()