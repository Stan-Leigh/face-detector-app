import numpy as np
import streamlit as st
import cv2 as cv
from PIL import Image

st.write("""
# Face Detector App
This app takes in an image and tries to identify all the faces in the image.
""")

st.sidebar.header('User Input')

# Collect image from the user
uploaded_file = st.sidebar.file_uploader("Upload your image in .jpg format", type=["jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    cv.imwrite('output.jpg', cv.cvtColor(img_array, cv.COLOR_RGB2BGR))

    img = cv.imread('output.jpg')
else:
    img = cv.imread('lady.jpg')

# # create a function to resize the image so we can show it properly.
# def rescaleFrame(frame, scale=0.75):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)

#     dimensions = (width, height)

#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# img = rescaleFrame(img)

# convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Haar cascades are really sensitive to noise in an image
# Anything that looks like a face, haar cascade will detect it even if it isn't a face.
# You can fine tune haar cascade to detect faces better by changing the minNeighbors and scaleFactor parameters
haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

num_of_faces = len(faces_rect)

st.markdown("## Number of faces detected")
st.write('There are', num_of_faces, 'face(s) in this image')

for (x,y,w,h) in faces_rect:
    x1, y1 = x+w, y+h

    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    # Top left design (x,y)
    cv.line(img, (x, y), (x+30, y), (0,255,0), 5)
    cv.line(img, (x, y), (x, y+30), (0,255,0), 5)

    # Top right design (x1,y)
    cv.line(img, (x1, y), (x1-30, y), (0,255,0), 5)
    cv.line(img, (x1, y), (x1, y+30), (0,255,0), 5)

    # Bottom left design (x,y1)
    cv.line(img, (x, y1), (x+30, y1), (0,255,0), 5)
    cv.line(img, (x, y1), (x, y1-30), (0,255,0), 5)

    # Bottom right design (x1,y1)
    cv.line(img, (x1, y1), (x1-30, y1), (0,255,0), 5)
    cv.line(img, (x1, y1), (x1, y1-30), (0,255,0), 5)

st.markdown("## Image display")
st.image(img, channels='BGR', caption='Image')