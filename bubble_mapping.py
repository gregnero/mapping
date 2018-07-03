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
from skimage import util
from matplotlib.patches import Circle


#Read in the csv bubble data as a DataFrame. Change directory as needed.
bubble_data = pd.read_csv("../Desktop/mapping_data/bubbly.csv")

#Convert the DataFrame into a numpy array
bubble_data = bubble_data.values

#Create array that contains the IDs
IDs = bubble_data[:,0].reshape((3744,1))

#Create an array that contains the glon,glat coordinates in degrees and the
#effective radius in degrees
glon_glat_reff = (np.vstack((bubble_data[:,1],bubble_data[:,2],bubble_data[:,3]/60))).T

#Initialize a dictionary to store the bubble data
bubble_dict = {}

#Create the dictionary from the arrays above
for i in range(0,3744):
    bubble_dict[IDs[i,0]] = (glon_glat_reff[i,0],glon_glat_reff[i,1],glon_glat_reff[i,2])

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


def degree_to_index(glon, glat, reff, array):
    '''
    Returns array index tuple cooresponding to the glon, glat values
    And a converted effective radius pixel value
    
    Args:
    --------------
    glon: Floating. Galactic longitude. Unit: degrees. Expects range: [0:360]
    glat: Floating. Galactic latitude. Unit: degrees. Expects range: [-1:1]
    reff: floating. Unit: degrees.
    array: a numpy ndarray
    
    Usage: index = degree_to_index(220.34, -0.4, 0.0348, my_image_array)

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


#Initialize bubble dictionary for storing the id and its respective x,y array location
#along with the effective radius in pixels
bubble_dict_idx = {}

#Convert degree values to array index values
for ID, degree_tuple in bubble_dict.items():
    bubble_dict_idx[ID] = degree_to_index(degree_tuple[0],degree_tuple[1],
                                          degree_tuple[2], final_panorama)

#Put it all together!!!! Plot the image with the bubbles on it
fig,ax = plt.subplots()
ax.imshow(final_panorama)
ax.set_aspect('equal')
for pixel_tuple in bubble_dict_idx.values():
    ax.add_patch(Circle((pixel_tuple[0],pixel_tuple[1]),pixel_tuple[2],
                         facecolor='none',edgecolor='b'))
plt.show()


