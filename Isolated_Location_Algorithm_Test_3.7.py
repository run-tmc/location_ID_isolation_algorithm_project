# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 20:24:37 2017

@author: Trevor M. Clark
"""
"""This algorithm computes the most isolated latitude and longitude location
   in a dataset.  The location isolation will be quantified by sum of the
   distances from a given latitude/longitude location to other the 
   latitude/longitude locations in the dataset.  The location ID with greatest
   distance sum will be considered the most isolated location"""
   
import numpy as np
import pandas as pd
from time import time


"""Import Dataset as a Pandas Dataform"""

program_dataset_file = "Data_Scientist_Product.csv"

print (program_dataset_file)

location_id_dataset_unprocessed = pd.read_csv(program_dataset_file)

print
print (location_id_dataset_unprocessed.head()) 

"""Dataset Pre-processing"""

# Determine the number of locations in the dataset based on the length
# of the 'Country' column
dataset_row_length = len(location_id_dataset_unprocessed['Country'])

print ("Index length", dataset_row_length)

# Make list of column names

list_column_names = location_id_dataset_unprocessed.columns.values.tolist()

# Checkpoint placeholder print "Column names",list_column_names 

"""Search and remove rows with missing (null) data"""

# Initial update status flag variable
i=0
for col_names in list_column_names:
    for row_index in range(dataset_row_length):
        if (pd.isnull(location_id_dataset_unprocessed.loc[row_index][col_names])):
            print ("Dropping Row", row_index, "due to NaN in",col_names, "column")
            i+=1
            
# Check if dataset updates are needed based on the update status flag value            

if (i != 0):
    location_id_dataset_unprocessed.dropna(axis=0, how='any', inplace=True)
    print ("Updated Dataset")
    print (location_id_dataset_unprocessed.head())
    
    # Save updated dataset as processed csv file
    location_id_dataset_unprocessed.to_csv('processed_dataset.csv')
    
    #Read back processed
    location_id_dataset = pd.read_csv("processed_dataset.csv")

    list_column_names = location_id_dataset.columns.values.tolist()

    # Check point placeholder print "Read in Column names",list_column_names 

    location_id_dataset.drop(list_column_names[0], axis=1, inplace=True)
    print
    print ("Read in Processed Dataset")
    print (location_id_dataset.head())
    
    # Determine the number of locations in the dataset based on the length
    # of the 'Country' column after post processing
    dataset_row_length = len(location_id_dataset['Country'])
    print
    print ("Updated Index length", dataset_row_length)


else:
    # No change to the dataset nor with the dataset row_length
    location_id_dataset = pd.read_csv(program_dataset_file)


# Add 'Distance_Sum Column to dataset and initialize all values to zero'
location_id_dataset['Distance_Sum'] = 0.0

print
print ("Dataset for Isolated Location Processing")
print (location_id_dataset.head())

"""This function computes the great cirle distance (in non-dimensional units) between 
   two locations from their respective latitude (lat) and longitude (lon) 
   coordinates using the Haversine formulas. The location lat and lon 
   coordinates are assumed to be in degrees and decimal degrees with negative 
   latitude degrees being in the south of the equator and negative longitude 
   degrees being the west of the prime meridan"""
   
def lat_lon_distance(lat1,lon1,lat2,lon2):
    # Error check location latitudes and longitudes
    if (abs(lat1)>90 or abs(lat2)>90 or abs(lon1)>180 or abs(lon2)>180):
        print ("Location Latitudes and Longitude not valid")
        return None
    # Compute distance for validate location latitudes and longitudes
    else:
        # Average Earth radius in non-dimensional units
        earth_radius = 1.0
        
        # Convert All lat and lon to radians
        lat1_rads = np.deg2rad(lat1)
        lon1_rads = np.deg2rad(lon1)
        lat2_rads = np.deg2rad(lat2)
        lon2_rads = np.deg2rad(lon2)
        
        # Compute delta lat and lon in radians
        delta_lat_rads = lat2_rads-lat1_rads
        delta_lon_rads = lon2_rads-lon1_rads
        
        a = (np.sin(delta_lat_rads/2.0)*np.sin(delta_lat_rads/2.0))\
        + np.cos(lat1_rads)*np.cos(lat2_rads)\
        * (np.sin(delta_lon_rads/2.0)*np.sin(delta_lon_rads/2.0))
        
        c = 2.0 * np.arctan2(np.sqrt(a),np.sqrt(1.0 - a))
        
        distance = earth_radius*c
        return distance
           
"""This function computes the great cirle distance (in non-dimensional units) 
   sum between for a given latitude (lat) and longitude (lon) and the other 
   locations in the dataset and saves the result in respective 
   location's "Distance_Sum" """
   
def compute_location_distance_sum(loc_id_ds,passed_index):
    # Initialize dist_sum to zero
    dist_sum = 0
    # Loop through each location
    for index2 in range(len(loc_id_ds['Country'])):
        # print "Index 2", index2
        # Only compute distances for different locations and for locations in 
        # different countries (assumes location distances within a country 
        # won't be major factors for determining isolation) 
        if ((passed_index != index2) and\
           (loc_id_ds.loc[passed_index]['Country'] != loc_id_ds.loc[index2]['Country'])):
            dist_sum +=\
            lat_lon_distance(loc_id_ds.loc[passed_index]['Latitude'],\
            loc_id_ds.loc[passed_index]['Longitude'],\
            loc_id_ds.loc[index2]['Latitude'],\
            loc_id_ds.loc[index2]['Longitude'])
        
    return dist_sum

print
print ("Index length", len(location_id_dataset['Country'])) 

start = time()
# Find the Distance Sums for each location
for index1 in range(len(location_id_dataset['Country'])):
    # print "Index1", index1
    location_id_dataset.set_value(index1,'Distance_Sum',\
    compute_location_distance_sum(location_id_dataset,index1))

end = time()
        
sorted_dataset = location_id_dataset.sort_values(by='Distance_Sum', axis=0,\
                                                 ascending=False)
print                                                 
print ("Finished calculations in {:.4f} seconds.".format(end - start))

print
print ("Sorted Isolated Location Based on Distance Sums")
print (sorted_dataset.head())                                               

"""Print the most isolated LocationId from the first row in the sorted dataset""" 

print                                                
print ("The most isolated LocationId is ",sorted_dataset.iloc[0]['LocationId'])

"""Determine most isolated LocationId in each country"""

# Initialize country_list as list data type
country_list = []

# Establish list of country name from the sorted dataset dataframe 
# 'Country' Column
country_list = sorted_dataset['Country'].drop_duplicates().tolist()

# Checkpoint placeholder print "Non-Dup Country List", country_list

# Find and print the most isolated locationId for each country by searching for
# first country name result in the sorted dataset dataform
print
for country_names in country_list:
    i=0
    while (country_names != sorted_dataset.iloc[i]['Country']):
        i+=1
    else:
        print ("LocationId",sorted_dataset.iloc[i]['LocationId'],\
        " is the most isolated location in Country ",sorted_dataset.iloc[i]\
                                                                   ['Country'])
        
            
            
    



    
    
        

