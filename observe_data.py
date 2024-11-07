import os
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA


# Define the path to the main data folder: code will iterate trough relvant files
main_data_folder = './data/img_test'    

#check if data path exists !
print(os.path.isdir(main_data_folder))

# Initialize the HSI dataset and define file extension: contains all paths of hdr and data files
dataset =HsiDataset(main_data_folder,data_ext='hyspex') #raw files from the camera have '.hyspex' extension , corrected files have '.ref' extension
nb_images = len(dataset)
print(f"dataset contains {nb_images} image(s)")

HSIreader = HsiReader(dataset)

# Loop through each hyperspectral image in the dataset
# for idx in range(len(dataset)):
#     HSIreader.read_image(idx)

#choose an image to process e.g. first img idx=0
idx=0   
HSIreader.read_image(idx)
print(f"read image{HSIreader.current_name}")

#define wavelenrghts (for plots mostly)
wv =HSIreader.get_wavelength()
wv = [int(l) for l in wv]

# Get the pseudo rgb imahe from hyperspectral data and plot image
pseudo_rgb= HSIreader.get_rgb()
plt.figure()
plt.imshow(pseudo_rgb)
plt.axis('off')
plt.show()

#check spectral data by manually selecting pixel-spectrum samples
HSIreader.get_spectrum();
