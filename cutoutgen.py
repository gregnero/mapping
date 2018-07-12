'''
Generate bubble-centric cutouts for the neural network.
Author: Gregory M. Nero
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import os
import sys
from skimage import io
from skimage.transform import resize, rescale

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


def dict_adjust(original_bubble_dict, dict_with_cutouts):
    '''
    Returns:
    ______________
    dictionary with form: {ID: bubble_cutout, glon, glat, r_eff, hitrate, ra, dec} or
                          {ID_control: control_cutout, X, X, X, X, X, X} 
    [0]image_array
    [1]glon: degrees
    [2]glat: degrees
    [3]r_eff: degrees
    [4]hitrate: unitless
    [5]ra: degrees
    [6]dec: degrees    
    
    Note: for control cutouts, the metadata is represented by placeholder X's (str)
          intend to add meaningful metadata later 
 
    Args:
    --------------
    original_bubble_dict: dict, the original dictionary of bubble:numerics data
    dict_with_cutouts: dict, the cutout/control dictionary with other metadata  

    Usage:
    ______________
    new_adjusted_dictionary = dict_adjust(bubble_dict, cutout_dict)

    '''
    
    #Initialize the new dictionary to return later
    new_dict = {}
    
    #Loop through the two dictionaries to match IDs and consolidate meaningful metadata
    for name1, value1 in original_bubble_dict.items():
        for name2, value2 in dict_with_cutouts.items():
            #This if catches the bubble cutouts
            if name1 == name2:
                new_dict[name1] = (value2[0], value1)
                break 
            #This elif catches the control cutouts
            elif (name1 + "_control") == name2:
                new_dict[name1 + "_control"] = (value2[0], 'X', 'X', 'X', 'X', 'X', 'X')
                break
            else:
                continue

    return new_dict


#Read in the bubble csv with pandas and then convert to a numpy array
bubble_data = (pd.read_csv("../Desktop/mapping_data/bubbly.csv")).values

#Create an array of the bubble IDs (columnated)
IDs = bubble_data[:,0].reshape((3744,1))

#Create an array with the quantitative bubble info (columnated)
'''Current numerics:
   [0]galactic longitude (degrees)
   [1]galactic latitude  (degrees)
   [2]effective radius   (degrees)
   [3]hitrate            (unitless)
   [4]ra                 (degrees)
   [5]dec                (degrees)
'''
bubble_numerics = (np.vstack((bubble_data[:,1], bubble_data[:,2],
                              bubble_data[:,3]/60, bubble_data[:,4],
                              bubble_data[:,5], bubble_data[:,6]))).T

#Initialize a dictionary to store the bubble data
bubble_dict = {}

#Fill the bubble dictionary with the IDs as keys and numerics as values
for i in range(0,3744):
    bubble_dict[IDs[i,0]] = (bubble_numerics[i,0], bubble_numerics[i,1],
                             bubble_numerics[i,2], bubble_numerics[i,3],
                             bubble_numerics[i,4], bubble_numerics[i,5])

if not 'image_dict' in globals():
    #Initialize an ordered dictionary to store image (name:array) pairs
    image_dict = collections.OrderedDict()

    #Load the sorted image names and arrays into the dictionary
    for file in sorted(os.listdir("../Desktop/mapping_data")):
        if file.endswith(".jpg"):
            with open(os.path.join("../Desktop/mapping_data", file), 'rb') as myfile:
                image_dict[file] = io.imread(myfile.name)
                print("Status: Loaded image %s" %file)

if not 'final_panorama' in globals():
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

'''Initialize dictionaries for storing the cutouts
>cutout_dict contains clean, valid bubbles with no boundary errors
>control_dict contains clean, valid control samples that coorespond to each bubble's mirror location
'''
cutout_dict = {}
control_dict = {}

'''Initialize lists to store names (only) of the skipped bubbles
>skipped_cutouts stores the names of the cutouts that were skipped
>skipped_controls stores the names of the controls that were skipped
'''
skipped_cutouts = []
skipped_controls = []

#Loop parameters
pad = 10
dim = (224,224,3)

#Loop to create the cutouts, control cutouts, and trash
for name,value in converted_bubble_dict.items():

    #Get values from converted_bubble_dict
    bubble_center = (value[0][0], value[0][1])
    radius = value[0][2]
    hitrate = value[1]
    
    #Get the center for each control, which is the mirror location of the bubble
    control_center = ((386999 - bubble_center[0]), (5999 - bubble_center[1])) 

    #Establish crop borders for the bubble
    bubble_left = bubble_center[0] - (radius + pad)
    bubble_right = bubble_center[0] + (radius + pad)
    bubble_top = bubble_center[1] - (radius + pad)
    bubble_bot = bubble_center[1] + (radius + pad)

    #Skip bubbles that leak outside of meaningful array range
    if bubble_top < 0 or bubble_bot > 5999 or bubble_left < 0 or bubble_right > 386999:
       skipped_cutouts.append(name)
       continue 
   
    #Establish crop borders for the control
    control_left = control_center[0] - (radius + pad)
    control_right = control_center[0] + (radius + pad)
    control_top = control_center[1] - (radius + pad)
    control_bot = control_center[1] + (radius + pad)

    #Skip controls that leak outside of meaningful array range
    if control_top < 0 or control_bot > 5999 or control_left < 0 or control_right > 386999:
       skipped_controls.append(name)
       continue 

    #Extract the bubble from the array
    extracted_bubble = final_panorama[bubble_top:bubble_bot,bubble_left:bubble_right,:]

    #Extract the control from the array
    extracted_control = final_panorama[control_top:control_bot,control_left:control_right,:]

    #Make the resized cutout of the bubble
    bubble_cutout = resize(extracted_bubble, dim)

    #Make the resized cutout of the control
    control_cutout = resize(extracted_control, dim)

    #Fill the cutout dictionary 
    cutout_dict[name] = (bubble_cutout, radius, hitrate, bubble_center)

    #Fill the control dictionary
    control_name = name + "_control"
    control_dict[control_name] = (control_cutout, radius, hitrate, control_center)



prepared_cutout_dict = dict_adjust(bubble_dict, cutout_dict)
prepared_control_dict = dict_adjust(bubble_dict, control_dict)


'''Prove mirror symmetry between bubble cutout and control
test_name = '1G303341-007180'
print(cutout_dict[test_name][3][0], cutout_dict[test_name][3][1])
print(control_dict[test_name][3][0], control_dict['1G303341-007180'][3][1])

plt.scatter(cutout_dict[test_name][3][0], cutout_dict[test_name][3][1])
plt.scatter(control_dict[test_name][3][0], control_dict[test_name][3][1])
plt.scatter(193499, 2999, marker='x')

plt.show()
'''

'''Prove that all the names match up between bubble cutouts and controls
successlist = []

for the_name in cutout_dict.keys():
    if (the_name in control_dict.keys()) == True:
        successlist.append(1)
print(len(list(cutout_dict.keys())))
print(len(list(control_dict.keys())))
print(len(successlist))
'''
