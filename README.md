# Face Detector App

A simple computer vision web application built with **Streamlit** and **OpenCV** that detects faces in an image and draws bounding boxes around each detected face.

The app allows users to either upload a `.jpg` image or capture an image using their device camera. It then converts the image to grayscale, applies a Haar Cascade face detector, counts the detected faces, and displays the image with highlighted face regions.

The deployed web app is live [here](https://stan-leigh-face-detector-app.streamlit.app/)

---

## Overview

The **Face Detector App** is an interactive image processing project that demonstrates how traditional computer vision techniques can be deployed through a simple web interface.

The app uses OpenCV's Haar Cascade classifier to identify faces in an input image. Users can test the app by uploading an image file or using the browser camera capture feature provided by Streamlit.

This project is useful for demonstrating:

- Basic computer vision workflows
- Image upload and camera input handling
- Face detection with OpenCV
- Streamlit deployment for image-based applications
- Drawing bounding boxes and visual annotations on images

---

## Features

- Upload `.jpg` images for face detection
- Capture images directly from the browser camera
- Convert uploaded images into OpenCV-compatible arrays
- Convert images to grayscale before detection
- Detect faces using a Haar Cascade XML classifier
- Count the number of detected faces
- Draw bounding rectangles around each detected face
- Add custom corner-line styling around detected face boxes
- Display the final annotated image in the Streamlit interface

---

## How the App Works

The app follows a simple computer vision pipeline:

1. The user chooses an image input method:
   - Upload
   - Camera

2. The app reads the image.

3. The image is converted into an OpenCV-compatible format.

4. The image is converted to grayscale.

5. The Haar Cascade face detector is loaded from:

```text
haar_face.xml
```

6. The detector searches for faces using:

```python
faces_rect = haar_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=3
)
```

7. The number of detected faces is displayed.

8. Green rectangles and corner lines are drawn around detected faces.

9. The final image is displayed in the app.

---

## Input Methods

### 1. Image Upload

The app allows users to upload a `.jpg` file through the sidebar:

```python
uploaded_file = st.sidebar.file_uploader(
    "Upload your image in .jpg format",
    type=["jpg"]
)
```

When a file is uploaded, it is opened using Pillow, converted to a NumPy array, saved temporarily, and read with OpenCV.

---

### 2. Camera Input

The app also supports camera capture using Streamlit's `st.camera_input()`:

```python
image = st.camera_input("Capture Image")
```

The captured image is decoded into an OpenCV image array before face detection is applied.

---

### 3. Default Image Fallback

If no uploaded image is provided and the camera option is not active, the app attempts to load:

```text
lady.jpg
```

This file must be present in the project directory if you want the default fallback image to work.

---

## Face Detection Method

The project uses a Haar Cascade classifier:

```python
haar_cascade = cv.CascadeClassifier("haar_face.xml")
```

Haar Cascades are traditional computer vision classifiers that detect objects based on patterns of light and dark image regions. In this project, the classifier is used to detect human faces.

The detection settings are:

```python
scaleFactor=1.1
minNeighbors=3
```

### Parameter Meaning

| Parameter | Meaning |
|---|---|
| `scaleFactor` | Controls how much the image size is reduced at each image scale |
| `minNeighbors` | Controls how many neighboring rectangles are required before a face is accepted |

Lower `minNeighbors` values may detect more faces but can also increase false positives. Higher values may reduce false positives but miss some faces.

---

## Project Structure

A typical project structure should look like this:

```text
face-detector-app/
│
├── face_detector_app.py
├── haar_face.xml
├── lady.jpg
├── requirements.txt
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `face_detector_app.py` | Main Streamlit application script |
| `haar_face.xml` | Haar Cascade XML file used for face detection |
| `lady.jpg` | Optional default image used when no image is uploaded |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## Requirements

Recommended modern dependency versions:

```text
streamlit==1.37.1
numpy==1.26.4
opencv-python-headless==4.10.0.84
Pillow==10.4.0
```

Older versions of Pillow may fail to install on Streamlit Cloud because they can require source compilation and system-level libraries.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the App Locally

Run:

```bash
streamlit run face_detector_app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## Limitations

- Haar Cascade detection can be sensitive to lighting, pose, image quality, and occlusion.
- The detector may miss faces that are tilted, partially covered, or not front-facing.
- It may detect false positives in complex images.
- The current implementation only accepts `.jpg` uploads.
- The app does not currently support video face detection.
- The app does not identify who the person is; it only detects face-like regions.

---

## Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow

---

## Portfolio Summary

This project demonstrates:

- Computer vision application development
- Face detection with OpenCV
- Image upload and camera input handling
- Real-time visual annotation of detected objects
- Deployment of an image processing app with Streamlit
- Practical use of traditional machine learning/computer vision methods

---

## License

This project is for educational and portfolio purposes.
