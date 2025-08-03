# 🏢 Smart Office Object Detection System

A real-time object detection system designed specifically for smart office environments using YOLOv8 deep learning technology. This application can identify and locate common office objects including people, laptops, chairs, phones, and more from CCTV footage or uploaded images.

## 🎥 Demo Video

[\*\*📺 Watch the Demo ](https://drive.google.com/file/d/1WqyaKyNfI4AgLVIcGNt5j3UP3SRcBH25/view?usp=drivesdk)

## ✨ Features

- **Real-time Object Detection**: Powered by YOLOv8 for fast and accurate detection
- **Office-Specific Model**: Custom trained model optimized for office environments
- **6 Object Categories**: Detects person, laptop, chair, phone,monitor,keyboard
- **User-Friendly Interface**: Clean Streamlit web application
- **Image Upload Support**: Easy drag-and-drop image uploading
- **Bounding Box Visualization**: Clear visual representation of detected objects
- **Confidence Scores**: Shows detection confidence for each identified object
- **CCTV Compatible**: Works with security camera footage and internal office images

## 🛠️ Tech Stack

- **Python 3.11.7**- Core programming language
- **Streamlit** - Web application framework
- **Ultralytics YOLOv8** - Object detection model
- **Roboflow** - Dataset management and model training
- **OpenCV** - Image processing
- **PIL (Pillow)** - Image handling
- **NumPy** - Numerical computations

## 🚀 Quick Start

### Prerequisites

- Python 3.11.7
- pip package manager

🛠️ Python Environment Setup (Python 3.11.7)

This project was built and tested with Python 3.11.7. Please install this exact version before running the code.

Windows

# Step 1: Download the Python 3.11.7 installer
Start-Process https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe

# Step 2: Run the installer manually and during setup, make sure to:
#   ✅ Check "Add Python 3.11 to PATH"
#   ✅ Optionally choose "Install for all users"

# Step 3: After installation, verify the Python version
python --version


macOs 

# Step 1: Install Homebrew if you haven't already
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)

# Step 2: Install pyenv (a tool to manage multiple Python versions)
brew install pyenv

# Step 3: Install Python 3.11.7 using pyenv
pyenv install 3.11.7

# Step 4: Set Python 3.11.7 as the global default version
pyenv global 3.11.7

# Step 5: Verify the Python version
python --version


### Installation & Run

```bash
# Clone the repository
git clone https://github.com/SXbro/smart-office-detection1.git
cd smart-office-detection1/smart_office

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

**One-line setup after cloning:**

```bash
pip install -r requirements.txt && streamlit run app.py
```

## 🎯 Supported Object Classes

The model is trained to detect the following office objects:

1. **Person** - Office personnel and visitors
2. **Laptop** - Laptops and notebooks
3. **Chair** - Office chairs and seating
4. **Phone** - Mobile phones and desk phones
5. **Monitor** - Computer monitors and displays
6. **Keyboard** - Keyboards that located in front of the monitor 

## 🔮 Future Enhancements

- Improving the model metrics by training and adding more images to the dataset

## ⚠️ Limitations

- Model performance depends on image quality and lighting conditions
- Best results achieved with clear, well-lit office environments
- Processing time varies based on image resolution and system specs
- Currently supports static image analysis only
- It may not detects the objects in a high quality bc of the lack of training dataset

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Smart Office Detection

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 👨‍💻 Author

- Email: wupapo@gmail.com

## 🙏 Acknowledgments

- **Roboflow** - For providing excellent dataset management and model training platform
- **Ultralytics** - For the powerful YOLOv8 implementation
- **Streamlit** - For the intuitive web app framework
- **Smart Office Hackathon** - For inspiring this innovative project

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
