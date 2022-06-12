# *****************************************************
#
# Project : App Flask - Reconnaissance d'image Qualité
# Béton
# Api request
# Project 6 - LH
# Auteur : Ali Naama
#
# *****************************************************



import requests
import numpy as np
from PIL import Image

KERAS_REST_API_URL = "http://localhost:5000/predictapi"
IMAGE_PATH = "DatasetP6/Test/stone/nidcailloux001.jpg"

# load the input image
image = open(IMAGE_PATH, "rb").read()


# construct the payload for the request
payload = {"image": image}

# submit the request
req = requests.post(KERAS_REST_API_URL, files=payload).json()

# Get the maximum element from a Numpy array
arr = np.array(req)
print(arr)
#max_value = np.max(arr)
#print(max_value)
# Get the indices of maximum element in numpy array
#result = np.where(arr == np.amax(arr))
#print('Returned tuple of arrays :', result)

print(req)