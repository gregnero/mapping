'''
Generate bubble-centric cutouts for the neural network.
Author: Gregory M. Nero
'''

import pandas as pd
import numpy as np
import collections
import os
import sys
from skimage import io
import matplotlib.pyplot as plt

from skimage.transform import resize, rescale

#Read in the bubble csv with pandas and then convert to a numpy array
bubble_data = (pd.read_csv("../Desktop/mapping_data/bubbly.csv")).values

#Create an array of the bubble IDs (columnated)
IDs = bubble_data[:,0].reshape((3744,1))

#Create an array with the quantitative bubble info (columnated)
'''Current numerics:
   galactic longitude (degrees)
   galactic latitude  (degrees)
   effective radius   (degrees)
   hitrate            (unitless)
'''
bubble_numerics = (np.vstack((bubble_data[:,1], bubble_data[:,2],
                              bubble_data[:,3]/60, bubble_data[:,4]))).T

#Initialize a dictionary to store the bubble data
bubble_dict = {}

#Fill the bubble dictionary with the IDs as keys and numerics as values
for i in range(0,3744):
    bubble_dict[IDs[i,0]] = (bubble_numerics[i,0], bubble_numerics[i,1],
                             bubble_numerics[i,2], bubble_numerics[i,3])

#Initialize an ordered dictionary to store image (name:array) pairs
image_dict = collections.OrderedDict()

#Load the sorted image names and arrays into the dictionary
for file in sorted(os.listdir("../Desktop/mapping_data")):
    if file.endswith(".jpg"):
        with open(os.path.join("../Desktop/mapping_data", file), 'rb') as myfile:
            image_dict[file] = io.imread(myfile.name)
            print("Status: Loaded image %s" %file)

#Initialize a list to store all of the image arrays from image_dict
image_list_unordered = []

#Fill the list with the arrays
for array_value in image_dict.values():
    image_list_unordered.append(array_value)

#Split this list into the northgrid/southgrid lists
northgrid_list = image_list_unordered[0:22][::-1]
southgrid_list = image_list_unordered[22:43][::-1]

#Construct the spatially oriented northgrid and southgrid arrays
northgrid = np.concatenate(northgrid_list, axis=1)
print("Status: northgrid created.")
southgrid = np.concatenate(southgrid_list, axis=1)
print("Status: southgrid created.")

#Construct the final array of concatenated spatially arranged images
final_panorama = np.concatenate((northgrid,southgrid), axis=1)
print("Status: final image created.")

def degree_to_pixel(glon, glat, reff, array):
    '''
    Returns:
    ______________
    tuple: (glon_idx, glat_idx, radius_in pixels)
 
    Args:
    --------------
    glon: Floating. Galactic longitude. Unit: degrees. Expects range: [0:360]
    glat: Floating. Galactic latitude. Unit: degrees. Expects range: [-1:1]
    reff: floating. Unit: degrees.
    array: a numpy ndarray
    
    Usage:
    ______________
    index = degree_to_pixel(220.34, -0.4, 0.0348, my_image_array)

    '''

    #Make sure input is good
    if glon < 0 or glon > 360:
        print('arg error: glon is out of acceptable range')
    if glat < -1 or glat > 1:
        print('arg error: glat is out of acceptable range')
    if reff < 0:
        print('arg error: radius must be positive')

    #Regulate glon input to make the spherical wraparound less annoying
    if glon <= 360 and glon >= 295.5:
        glon = glon - 360

    #Get array shape
    rows = array.shape[0]
    cols = array.shape[1]

    #Set up the degree range for glon and glat based on the array shape
    glat_range = np.linspace(1,-1,rows)
    glon_range = np.linspace(64.5, -64.5, cols)

    #Get the index based on closes value
    glat_idx = (np.abs(glat_range - glat)).argmin()
    glon_idx = (np.abs(glon_range - glon)).argmin()

    #Get the effective pixel radius
    radius_in_pixels = int(reff * 3000)

    return (glon_idx, glat_idx, radius_in_pixels)

#Initialize dictionary to store same bubble IDs with their new converted values
converted_bubble_dict = {}

#Fill the dictionary with the converted values using degree_to_pixel
#Also add the numeric bubble values that don't need to be converted (currently only hitrate)
'''Dictionary has form:
   ID: ((glon_idx, glat_idx, radius_in_pixels), hitrate)
'''
for bubblename, numeric_tuple in bubble_dict.items():
    converted_bubble_dict[bubblename] = (degree_to_pixel(numeric_tuple[0], numeric_tuple[1],
                                                        numeric_tuple[2], final_panorama),
                                                        numeric_tuple[3])


cutout_dict = {}
abandoned_cutouts = {}

pad = 10
dim = (224,224,3)



for name,value in converted_bubble_dict.items():

    center = (value[0][0], value[0][1])
    radius = value[0][2]
    hitrate = value[1]
    
    left = center[0] - (radius + pad)
    right = center[0] + (radius + pad)
    top = center[1] - (radius + pad)
    bot = center[1] + (radius + pad)

    extracted_bubble = final_panorama[top:bot,left:right,:]

    if extracted_bubble.size == 0:
        abandoned_cutouts[name] = (extracted_bubble)
        continue

    cutout = resize(extracted_bubble, dim)

    cutout_dict[name] = (cutout, radius, hitrate)

'''Plotting
plt.imshow(final_panorama[:,0:9000,:])
plt.show()                                            
'''


