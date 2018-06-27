"""

Map the large bubble data discussed in the paper presented by Simpson et al. to a given set of images of the galactic plane captured by Spitzer.

"""

import pandas as pd
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys


#Read in the csv bubble data as a DataFrame. Change directory as needed.
bubble_data = pd.read_csv("../Desktop/mapping_data/bubbly.csv")

#Convert the DataFrame into a numpy array
bubble_data = bubble_data.values

#Initialize list to store image arrays
spitzer_images = []

#Load the image arrays into the list
for file in sorted(os.listdir("../Desktop/mapping_data")):
    if file.endswith(".jpg"):
        with open(os.path.join("../Desktop/mapping_data", file), 'rb') as myfile:
            spitzer_images.append(io.imread(myfile.name))
            print("Status: Loading image %s" %file)

