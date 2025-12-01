# OpenCV Tutorial Project

A comprehensive Python project demonstrating various OpenCV (Open Source Computer Vision Library) capabilities, from basic image operations to real-time face detection.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Modules & Sections](#modules--sections)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)
- [License](#license)

---

## 🎯 Overview

This project provides hands-on examples of OpenCV functionality in Python, covering:
- Basic image processing (read, display, resize, crop, rotate, flip)
- Drawing shapes and adding text overlays
- Real-time webcam capture and processing
- Image filtering and blurring techniques
- Edge detection and thresholding
- Contour detection and shape recognition
- Real-time face, eye, and smile detection using Haar Cascades

Perfect for beginners learning computer vision and intermediate developers looking for reference implementations.

---

## ✨ Features

### Core Image Operations
- ✅ Read and write images from disk
- ✅ Convert between color spaces (BGR ↔ Grayscale)
- ✅ Resize, crop, rotate, and flip images
- ✅ Interactive image processing with user input

### Drawing & Visualization
- ✅ Draw lines, rectangles, and circles
- ✅ Add custom text with various fonts and colors
- ✅ Overlay graphics on images

### Advanced Processing
- ✅ Gaussian and Median blur filters
- ✅ Custom kernel-based image sharpening
- ✅ Canny edge detection
- ✅ Binary thresholding with Otsu's method
- ✅ Bitwise operations (AND, OR, XOR, NOT)

### Computer Vision
- ✅ Contour detection and visualization
- ✅ Automated shape classification (Triangle, Square, Circle, etc.)
- ✅ Real-time face detection using Haar Cascades
- ✅ Eye and smile detection within detected faces
- ✅ Region of Interest (ROI) based processing

---

## 📁 Project Structure

```
D:\Desktop\OpenCV\
│
├── opencv.py                          # Main tutorial script (11 sections)
├── face_detection.py                  # Standalone face detection demo
├── face_eye_smile.py                  # Face, eye, and smile detection
│
├── requirements.txt                   # Python dependencies
├── README.MD                          # This file
│
├── haarcascade_frontalcatface.xml    # Haar Cascade for cat face detection
├── haarcascade_eye.xml               # Haar Cascade for eye detection
├── haarcascade_smile.xml             # Haar Cascade for smile detection
├── eye.xml                           # Additional eye detection cascade
│
└── venv/                             # Virtual environment (Python packages)
```

---

## 🔧 Prerequisites

- **Python**: 3.7 or higher
- **Operating System**: Windows, macOS, or Linux
- **Webcam**: Required for real-time detection features (optional for image processing)
- **Image Files**: Sample images (e.g., `Arnab.jpg`, `monalisha.png`, `OIP.jpg`) for testing

---

## 📦 Installation

### 1. Clone or Download the Project

```bash
cd D:\Desktop\OpenCV
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> **Note**: If PowerShell blocks script execution, run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- `opencv-python` - OpenCV library for Python
- `numpy` - Numerical computing library (required by OpenCV)

### 4. Verify Installation

```bash
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

Expected output: `OpenCV version: 4.x.x`

---

## 🚀 Usage

### Running the Main Tutorial

The `opencv.py` file contains 11 comprehensive sections. Run it with:

```bash
python opencv.py
```

**Interactive Sections:**
- Some sections prompt for user input (e.g., image paths, actions)
- Webcam sections require pressing `q` to quit
- Each image display waits for any key press to continue

### Running Face Detection Demos

**Basic Face Detection:**
```bash
python face_detection.py
```

**Face, Eye & Smile Detection:**
```bash
python face_eye_smile.py
```

**Controls:**
- Press `q` to quit the webcam window
- Ensure your webcam is not being used by another application

---

## 📚 Modules & Sections

### `opencv.py` - Complete Tutorial (11 Sections)

| Section | Description | Key Functions |
|---------|-------------|---------------|
| **1. Basic Image Operations** | Read, display, convert to grayscale, resize, save | `cv2.imread()`, `cv2.cvtColor()`, `cv2.resize()`, `cv2.imwrite()` |
| **2. Interactive Processing** | User-driven image operations | Input handling, conditional processing |
| **3. Image Transformations** | Resize, crop, rotate, flip operations | `cv2.warpAffine()`, `cv2.flip()`, `cv2.getRotationMatrix2D()` |
| **4. Drawing Shapes & Text** | Lines, rectangles, circles, text overlays | `cv2.line()`, `cv2.rectangle()`, `cv2.circle()`, `cv2.putText()` |
| **5. Webcam Capture** | Real-time video feed from camera | `cv2.VideoCapture()`, frame reading loop |
| **6. Image Filtering** | Gaussian blur, median blur, sharpening | `cv2.GaussianBlur()`, `cv2.medianBlur()`, `cv2.filter2D()` |
| **7. Edge Detection** | Canny algorithm, binary thresholding | `cv2.Canny()`, `cv2.threshold()` |
| **8. Bitwise Operations** | Logical operations on images | `cv2.bitwise_and()`, `cv2.bitwise_or()`, `cv2.bitwise_xor()`, `cv2.bitwise_not()` |
| **9. Contour Detection** | Finding and drawing object contours | `cv2.findContours()`, `cv2.drawContours()` |
| **10. Shape Detection** | Automated shape classification | `cv2.approxPolyDP()`, `cv2.contourArea()`, `cv2.minEnclosingCircle()` |
| **11. Face/Eye/Smile Detection** | Real-time detection with Haar Cascades | `cv2.CascadeClassifier()`, `detectMultiScale()` |

### `face_detection.py` - Standalone Face Detection

Simple webcam-based face detection using Haar Cascades. Features:
- Loads cascade classifier
- Captures webcam feed
- Detects faces in real-time
- Draws rectangles around detected faces

### `face_eye_smile.py` - Advanced Facial Feature Detection

Enhanced detection including:
- Face detection
- Eye detection within face regions (ROI)
- Smile detection within face regions
- Text labels for detected features

---

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### **1. Import Error: `ModuleNotFoundError: No module named 'cv2'`**

**Solution:**
```bash
# Activate virtual environment first
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # macOS/Linux

# Install OpenCV
pip install opencv-python
```

#### **2. Webcam Not Opening (`"Error: Could not open webcam"`)**

**Causes & Solutions:**
- **Camera in use**: Close other applications using the webcam (Zoom, Teams, etc.)
- **Permission denied**: Grant camera permissions in system settings
- **Wrong camera index**: Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`

#### **3. Cascade Classifier Not Loading (`cascade.empty() returns True`)**

**Solution:**
```python
# Verify file exists
import os
print(os.path.exists(r"D:\Desktop\OpenCV\haarcascade_frontalcatface.xml"))

# Use absolute paths with raw strings
cascade = cv2.CascadeClassifier(r"D:\Desktop\OpenCV\haarcascade_frontalcatface.xml")
```

#### **4. Image Not Found (`image is None`)**

**Solution:**
- Check file path is correct (use raw strings: `r"D:\path\to\image.jpg"`)
- Verify image file exists in the specified location
- Ensure image format is supported (JPG, PNG, BMP, TIFF)

#### **5. `cv2.error: (-215:Assertion failed) !_src.empty()`**

**Cause**: Trying to process an empty/invalid image

**Solution:**
```python
# Always check if image loaded successfully
image = cv2.imread("image.jpg")
if image is None:
    print("Failed to load image")
else:
    # Process image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

#### **6. Windows Not Closing After `cv2.imshow()`**

**Solution:**
```python
# Always include waitKey and destroyAllWindows
cv2.imshow("Window", image)
cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()  # Clean up
```

---

## 📖 Resources

### OpenCV Documentation
- **Official Docs**: https://docs.opencv.org/
- **Python Tutorials**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- **API Reference**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

### Haar Cascade Files
- **Pre-trained Cascades**: https://github.com/opencv/opencv/tree/master/data/haarcascades
- Common cascades:
  - `haarcascade_frontalface_default.xml` - Human face detection
  - `haarcascade_eye.xml` - Eye detection
  - `haarcascade_smile.xml` - Smile detection

### Learning Resources
- **OpenCV Course**: https://opencv.org/courses/
- **PyImageSearch**: https://pyimagesearch.com/
- **Real Python - OpenCV**: https://realpython.com/opencv-python/

### Community
- **Stack Overflow**: Tag `[opencv] [python]`
- **OpenCV Forum**: https://forum.opencv.org/

---

## 🎓 Key Concepts Explained

### Color Spaces
- **BGR**: OpenCV's default (Blue, Green, Red) - opposite of RGB
- **Grayscale**: Single channel, values 0-255
- **HSV**: Hue, Saturation, Value - useful for color-based segmentation

### Image Coordinates
```
(0,0) -------- x (width) -------->
  |
  |
  y (height)
  |
  |
  v
```

### Haar Cascades
Pre-trained classifiers using Haar-like features for object detection:
- **Fast**: Real-time performance
- **Trained**: Uses machine learning on thousands of images
- **Scale-invariant**: Detects objects at different sizes
- **Parameters**:
  - `scaleFactor`: How much image size is reduced at each scale (e.g., 1.1 = 10%)
  - `minNeighbors`: Minimum neighbors for valid detection (higher = fewer false positives)

### Region of Interest (ROI)
Selecting a specific area of an image for focused processing:
```python
# Extract face region
roi = image[y:y+h, x:x+w]

# Process only the ROI
eyes = eye_cascade.detectMultiScale(roi)
```

---

## 🔐 License

This project is provided for educational purposes. OpenCV itself is released under the Apache 2 License.

**Haar Cascade Files**: Provided by OpenCV (Apache 2 License)

---

## 👤 Author

**Arnab**

---

## 🤝 Contributing

Feel free to:
- Add more examples
- Improve documentation
- Fix bugs
- Suggest new features

---

## 📝 Notes

- **Image Paths**: Update paths in scripts to match your local file structure
- **Webcam Index**: Default is `0`; change if you have multiple cameras
- **Performance**: Real-time detection performance depends on CPU speed and image resolution
- **Cascade Selection**: Use `haarcascade_frontalface_default.xml` for human face detection instead of cat faces

---

## 🚦 Quick Start Checklist

- [ ] Python 3.7+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] OpenCV version verified
- [ ] Sample images available (or update paths)
- [ ] Webcam connected and functional
- [ ] Cascade XML files in project directory

---

**Happy Coding! 🎉**

Explore the fascinating world of computer vision with OpenCV!