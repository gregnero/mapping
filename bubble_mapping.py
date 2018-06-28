"""

Map the large bubble data discussed in the paper presented by Simpson et al. to
a given set of images of the galactic plane captured by Spitzer.

"""
import os
import matplotlib
import sys
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io

#Read in the csv bubble data as a DataFrame. Change directory as needed.
bubble_data = pd.read_csv("../Desktop/mapping_data/bubbly.csv")

#Convert the DataFrame into a numpy array
bubble_data = bubble_data.values

#Initialize list to store image arrays
image_values = []

#Initialize list to store image names
image_names = []

#Initialize ordered dictionary to store name:value pairs
image_dict = collections.OrderedDict()

#Load the sorted image values and names into their respective lists
for file in sorted(os.listdir("../Desktop/mapping_data")):
    if file.endswith(".jpg"):
        image_names.append(file) 
        with open(os.path.join("../Desktop/mapping_data", file), 'rb') as myfile:
            image_dict[file] = io.imread(myfile.name)
            print("Status: Loaded image %s" %file)

#Initialize list to store all of the individual image arrays
image_list_unordered = []

#Put the image arrays from the dictionary of images into the initialized array
for value in image_dict.values():
    image_list_unordered.append(value)

#Split previously created list of all of the images into their respective northgrid/
#southgrid categories. They are considered "unsorted" because they don't yet align
#spatially
northgrid_unordered_list = image_list_unordered[0:22]
southgrid_unordered_list = image_list_unordered[22:43]

#Align the list of images in each grid spatially
northgrid_list = northgrid_unordered_list[::-1]
southgrid_list = southgrid_unordered_list[::-1]

#Construct the spatially oriented northgrid and southgrid arrays
northgrid = np.concatenate(northgrid_list, axis=1)
southgrid = np.concatenate(southgrid_list, axis=1)

#Stitch the northgrid and southgrid together to make a HUGE image
final_panorama = np.concatenate((northgrid,southgrid), axis=1)

