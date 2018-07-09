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

#Initialize dictionaries for storing the cutouts
#The leaked_bubble_cutout_dict store cutouts that had an index outside of valid array.
cutout_dict = {} #Contains 3738 for my case
leaked_bubble_cutout_dict = {} #Contains 6 for my case

#Loop parameters
pad = 10
dim = (224,224,3)

#Loop to create the cutouts
for name,value in converted_bubble_dict.items():

    #Get values from converted_bubble_dict
    center = (value[0][0], value[0][1])
    radius = value[0][2]
    hitrate = value[1]
    
    #Establish crop borders for the bubble
    left = center[0] - (radius + pad)
    right = center[0] + (radius + pad)
    top = center[1] - (radius + pad)
    bot = center[1] + (radius + pad)

    #Corrects for out-of-range negative top indices
    #For some reason, the bubbles were only out of range on the top...
    if top <= 0:
        top = 0
        extracted_leaking_bubble = final_panorama[top:bot,left:right,:]
        leaked_cutout = resize(extracted_leaking_bubble, dim)
        leaked_bubble_cutout_dict[name] = (leaked_cutout, radius, hitrate)
        continue
    
    #Extract the bubble from the array
    extracted_bubble = final_panorama[top:bot,left:right,:]

    #Make the resized cutout
    cutout = resize(extracted_bubble, dim)

    #Fill the cutout dictionary with valid arrays relavant numerics
    cutout_dict[name] = (cutout, radius, hitrate)

def show_cutout_samples(cutout_dictionary, show_best, num_samples=12):
    '''
    Returns:
    -------------
    Figure with cutout samples

    Args:
    -------------
    cutout_dictionary: dictionary; contains names as keys and numeric tuple as values
    show_best: bool; if True, will show assortment of high hitrate bubbles
                     if Fasle, will show random assortment of bubbles
    num_samples: int; number of samples to show. Currently fixed at 12 for convenience

    Usage:
    -------------
    show_cutout_samples(cutout_dict, show_best=True)

    ''' 
    
    #Initialize lists to store dictionary information
    cutout_names = []
    cutout_images = []
    cutout_radii = []
    cutout_hitrates = []

    #Set hitrate threshold value for show_best restriction
    #Note: make sure at least 12 valid bubbles can be found...
    threshold = 0.5
    
    #Fill the lists depending on value of show_best
    for bub_name, bub_num in cutout_dictionary.items():
        if show_best == True:
            if bub_num[2] >= threshold:
                cutout_names.append(bub_name)
                cutout_images.append(bub_num[0])
                cutout_radii.append(bub_num[1])
                cutout_hitrates.append(bub_num[2])
            else:
                continue
        else:
            cutout_names.append(bub_name)
            cutout_images.append(bub_num[0])
            cutout_radii.append(bub_num[1])
            cutout_hitrates.append(bub_num[2])

    #Set up the figure 
    fig = plt.figure()
    if show_best == True:
        title = str(num_samples) + " Samples with High Hitrate" 
    else:
        title = str(num_samples) + " Cutout Samples"
    fig.suptitle(title, fontsize=18, x=0.5, y=0.986)
    fig.subplots_adjust(hspace = 0.75)

    #Plot the sample cutouts
    for sample in range(1,13):
        fig.add_subplot(3,4,sample)
        plt.imshow(cutout_images[sample])
        plot_title = cutout_names[sample] + "\n" + "(hitrate = " + str(cutout_hitrates[sample]) + ")"
        plt.title(plot_title, fontsize=6)
    print("Exit figure to continue...")
    plt.show()


show_cutout_samples(cutout_dict, show_best=False)

show_cutout_samples(cutout_dict, show_best=True)


'''Correlation between radius and hitrate
radii = []
hr = []
for info in cutout_dict.values():
    radii.append(info[1])
    hr.append(info[2])

plt.scatter(radii,hr)
plt.show()
'''
