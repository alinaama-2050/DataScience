# *****************************************************
#
# Project : App Flask - Reconnaissance d'image Qualité
# Béton
# Api request
# Project 6 - POC
# Auteur : Ali Naama
#
# *****************************************************



import requests
import numpy as np
from PIL import Image

KERAS_REST_API_URL = "http://localhost:5000/predictapi"
IMAGE_PATH = "DatasetP6/Test/black/Tachesnoires3.jpg"

# load the input image
image = open(IMAGE_PATH, "rb").read()


# construct the payload for the request
payload = {"image": image}

# submit the request
req = requests.post(KERAS_REST_API_URL, files=payload).json()
print(req)
