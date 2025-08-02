#!/usr/bin/env python3
"""
Quick test script for Smart Office Challenge setup
"""

def test_imports():
    """Test all critical imports"""
    print("Testing Smart Office Challenge dependencies...")
    
    tests = [
        ("import numpy as np", "NumPy"),
        ("import pandas as pd", "Pandas"),
        ("import matplotlib.pyplot as plt", "Matplotlib"),
        ("from PIL import Image", "Pillow"),
        ("import cv2", "OpenCV"),
        ("import yaml", "PyYAML"),
        ("import torch", "PyTorch"),
        ("from ultralytics import YOLO", "Ultralytics"),
        ("import streamlit as st", "Streamlit"),
        ("import plotly.express as px", "Plotly")
    ]
    
    success = 0
    for test_code, name in tests:
        try:
            exec(test_code)
            print(f"✅ {name}")
            success += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    print(f"\n📊 Test Results: {success}/{len(tests)} packages working")
    
    if success >= 8:
        print("🎉 Setup successful! Ready to run Smart Office Challenge.")
        print("\n🚀 Next steps:")
        print("   python run_pipeline.py --complete")
        print("   streamlit run app.py")
    else:
        print("⚠️  Setup incomplete. Check error messages above.")

if __name__ == "__main__":
    test_imports()
