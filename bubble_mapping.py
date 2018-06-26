"""

Map the large bubble data discussed in the paper presented by Simpson et al. to a given set of images of the galactic plane captured by Spitzer.

"""

import pandas as pd
import numpy as np

#Read in the csv bubble data as a DataFrame. Change directory as needed.
bubble_data = pd.read_csv("../Desktop/mapping_data/bubbly.csv")

#Convert the DataFrame into a numpy array
bubble_data = bubble_data.values


