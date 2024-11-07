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