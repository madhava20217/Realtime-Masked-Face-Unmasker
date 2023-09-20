# %%
import glob
import os
import pathlib

import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import functional as F

from torchvision import transforms
from torchinfo import torchinfo
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
import torch.nn.functional as F

from PIL import Image

import torchmetrics
from torchvision.utils import save_image, make_grid
import cv2
# from util.io import load_ckpt

# from util.loss import  InpaintingLoss
import os, glob

import efficientunet
import random

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device("cpu")


import wandb

# %%
dataset_path = '../Datasets/inpainting/img_align_celeba/'

new_dataset_path = '../Datasets/inpainting/just_faces/'

# %%
files = glob.glob(dataset_path+'*.jpg')

# %%
sizes = (256, 256)

rescale_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(sizes, antialias= False)
])

# %%
img = np.array(Image.open(dataset_path+'000041.jpg'))

# %%
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img, 1.1, 2)

# loop over all the detected faces
for (x,y,w,h) in faces:
   face = img[y:y + h, x:x + w]

# Display an image in a window


# %%
face.shape

# %%
# plt.imshow(face)

# %%
def get_face(img):
    faces = face_cascade.detectMultiScale(img, 1.1, 2)
    for (x,y,w,h) in faces:
        face = img[y:y + h, x:x + w]
    return face

# %%
os.makedirs(new_dataset_path, exist_ok= True)

# %%
failures = 0
bar = tqdm(files)
for x in bar:
            try:
                face = get_face(np.array(Image.open(x)))
                faceshot = Image.fromarray(face)
                faceshot.save(os.path.join(new_dataset_path, os.path.basename(x)))
                # faces.append(rescale_transform(face))
            except:
                failures+=1
                bar.set_description_str("Failures: {}".format(failures))
