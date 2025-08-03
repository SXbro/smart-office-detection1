# 🏢 Smart Office Object Detection System

A real-time object detection system designed specifically for smart office environments using YOLOv8 deep learning technology. This application can identify and locate common office objects including people, laptops, chairs, phones, and more from CCTV footage or uploaded images.

## 🎥 Demo Video

[**📺 Watch the Demo**](https://drive.google.com/file/d/1WqyaKyNfI4AgLVIcGNt5j3UP3SRcBH25/view?usp=drivesdk)

## ✨ Features

- **Real-time Object Detection**: Powered by YOLOv8 for fast and accurate detection
- **Office-Specific Model**: Custom trained model optimized for office environments
- **6 Object Categories**: Detects person, laptop, chair, phone, monitor, keyboard
- **User-Friendly Interface**: Clean Streamlit web application
- **Image Upload Support**: Easy drag-and-drop image uploading
- **Bounding Box Visualization**: Clear visual representation of detected objects
- **Confidence Scores**: Shows detection confidence for each identified object
- **CCTV Compatible**: Works with security camera footage and internal office images

## 🛠️ Tech Stack

- **Python 3.10** - Core programming language
- **Streamlit** - Web application framework
- **Ultralytics YOLOv8** - Object detection model
- **Roboflow** - Dataset management and model training
- **OpenCV** - Image processing
- **PIL (Pillow)** - Image handling
- **NumPy** - Numerical computations

## 🚀 Quick Start

### Prerequisites

- Python 3.10
- pip package manager

## 🛠️ Python Environment Setup (Python 3.10)

This project was built and tested with Python 3.10. Please install this version before running the code.

### Windows

**Step 1: Download Python 3.10**
```powershell
# Open PowerShell as Administrator and run:
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.10.12/python-3.10.12-amd64.exe" -OutFile "python-3.10.12-amd64.exe"
```

**Step 2: Install Python**
```powershell
# Run the installer with automatic setup:
.\python-3.10.12-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
```

**Step 3: Verify Installation**
```powershell
# Close and reopen PowerShell, then verify:
python --version
```

**Alternative Manual Installation:**
1. Download: https://www.python.org/ftp/python/3.10.12/python-3.10.12-amd64.exe
2. Run installer and check "Add Python 3.10 to PATH"
3. Verify with `python --version`

### macOS

**Step 1: Install Homebrew (if not already installed)**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Step 2: Install pyenv**
```bash
brew install pyenv
```

**Step 3: Install Python 3.10**
```bash
pyenv install 3.10.12
```

**Step 4: Set as global version**
```bash
pyenv global 3.10.12
```

**Step 5: Update shell configuration**
```bash
# For zsh (default on macOS):
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# For bash:
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
source ~/.bash_profile
```

**Step 6: Verify Installation**
```bash
python --version
```

**Alternative Direct Installation:**
```bash
# Download and install directly:
curl -O https://www.python.org/ftp/python/3.10.12/python-3.10.12-macos11.pkg
sudo installer -pkg python-3.10.12-macos11.pkg -target /
```

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
