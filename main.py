# Import Necessary packages
import streamlit as st

from fastai.vision.all import *
from fastai.vision import models

import numpy as np
import cv2
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import albumentations

class AlbumentationsTransform(DisplayedTransform):
    split_idx,order=0,2
    def __init__(self, train_aug): store_attr()
    
    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)

# Image Augmentations
def get_train_augmentation(): return albumentations.Compose([
            albumentations.RandomCrop(224,224),
            albumentations.HorizontalFlip(p=0.25),
            albumentations.VerticalFlip(p=0.25),
            albumentations.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.3, p=0.25),
            albumentations.SafeRotate(limit=90, p=0.25),
])

from fastbook import load_learner
model = load_learner("./resnet34-okra.pkl")

def detect(img):
    image = img
    image = np.array(image)

    result = model.predict(image)
    img = image.copy()
    st.metric(label="Class Label", value=result[0])
    
    if result[0] == "Diseased Okra":
        st.error('Disease found in okra plant', icon="🚨")
    else:
        st.success('Healthy okra plant', icon="✅")

    return result[0]


def segment(img):
    image = np.array(img)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([45, 255, 255])

    mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

    masked_img = cv2.bitwise_and(image, image, mask=mask)
    st.image(masked_img, caption="Segmented Disease Image", width=400)

def main():        
    st.title("Okra Disease Detection Model")
    uploaded_file = st.file_uploader('Upload Input Image', type=['jpg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Image", clamp=True, width=400)

    if st.button('Detect', key=1):
        if uploaded_file is not None:
            with st.spinner("Processing..."):
                st.write("---") 
                st.subheader("Classification")
                result = detect(image)
                if result == "Diseased Okra":
                    st.write("---") 
                    st.subheader("Segmentation")
                    segment(image)
                    st.write("---") 
        else:
            st.write('Upload an image first!')
    else:        
        st.info('Browse for the input image and click on Detect Button')

if __name__ == '__main__':
    st.set_page_config(page_title="Okra Disease Detection Model", layout="wide")
    main()
