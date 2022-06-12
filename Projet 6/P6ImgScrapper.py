
###############################################################################
# Description : This program used for Project 6
# Get Image From Web Scrapping
# Author : ANA
# Date : 08/01/2022
###############################################################################

# Required Libs
import pandas as pd
import requests
import shutil
from shutil import copyfile
from urllib.request import urlopen
from bs4 import BeautifulSoup
from PIL import Image

# Web Site To Scrap

site = "https://www.holcimpartner.ch/fr/betonpraxis/colorations"
site = "https://www.holcim.be/fr/guide-pratique-causes-et-prevention-des-alterations-du-beton"
html = urlopen(site)
bs = BeautifulSoup(html, 'html.parser')
url_base = site
images = bs.find_all('img')
i = 0

# Image Extarction and flush into Hard Drive
for img in images:
    if img.has_attr('src'):
        print(img['src'])
        url_ext = img['src']
        full_url = url_base + url_ext  # Combining first 2 variables to create       a complete URLr = requests.get(full_url, stream=True) #Get request on full_urlif r.status_code == 200:                     #200 status code = OK
        print(full_url)
        with open("data/image/defaut-beton-" + str(i) + "_.png", 'wb') as f:
            try:
                r = requests.get(url_ext, stream=True)  # Get request on full_url
                r.raw.decode_content = True
                try:
                    shutil.copyfileobj(r.raw, f)
                    print("File copied successfully.")
                # If source and destination are same
                except shutil.SameFileError:
                    print("Source and destination represents the same file.")

            except:
                print("url not found " )

            i = i + 1
            try:
                print("Image.open")
                #image = Image.open("data/image/defaut-beton-" + str(i) + "_.png")
                #image.show()
            except:
                print("image display not available")
