import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances



def clean(df_cars_dirty):
    """
    Handle all "dirty" records in the cars dataframe

    Inputs:
    - df_cars_dirty: Pandas dataframe of dataset containing "dirty" records

    Returns:
    - df_cars_cleaned: Pandas dataframe of dataset without "dirty" records
    """   
    
    # We first create a copy of the dataset and use this one to clean the data.
    df_cars_cleaned = df_cars_dirty.copy()

    #########################################################################################
    ### Your code starts here ###############################################################

    # Convert to string just in case some values are mixed types
    
    # clean listing_id
    df_cars_cleaned = df_cars_cleaned[df_cars_cleaned['listing_id'].astype(str).str.isdigit()]
    print('Number of records after cleaning listing_id: {}'.format(len(df_cars_cleaned)))
    
    # clean invalid year and owner
    df_cars_cleaned = df_cars_cleaned[(df_cars_cleaned['manufactured'] < 2025) & (df_cars_cleaned['manufactured'] > 1900)]
    df_cars_cleaned = df_cars_cleaned[df_cars_cleaned['no_of_owners'] >= 1]
    print('Number of records after cleaning manufactured year and no_of_owners: {}'.format(len(df_cars_cleaned)))
    
    # clean type_of_vehicle
    df_cars_cleaned['type_of_vehicle'] = df_cars_cleaned['type_of_vehicle'].replace('unknown', np.nan)
    print('Number of records after cleaning type_of_vehicle: {}'.format(len(df_cars_cleaned)))
    
    
    ### Your code ends here #################################################################
    #########################################################################################
    
    return df_cars_cleaned


def handle_nan(df_cars_nan):
    """
    Handle all nan values in the cars dataframe

    Inputs:
    - df_cars_nan: Pandas dataframe of dataset containing nan values

    Returns:
    - df_cars_no_nan: Pandas dataframe of dataset without nan values
    """       

    # We first create a copy of the dataset and use this one to clean the data.
    df_cars_no_nan = df_cars_nan.copy()

    #########################################################################################
    ### Your code starts here ###############################################################
    
    # remove url column
    df_cars_no_nan = df_cars_no_nan.drop(columns=['url'])
    
    # replace missing brand with 'unknown'
    df_cars_no_nan['make'] = df_cars_no_nan['make'].fillna('unknown')
    
    # replace missing mileage with median mileage
    df_cars_no_nan['mileage'] = df_cars_no_nan['mileage'].fillna(df_cars_no_nan['mileage'].median())
    
    # drop rows with missing price
    df_cars_no_nan = df_cars_no_nan.dropna(subset=['price'])
    

    ### Your code ends here #################################################################
    #########################################################################################
    
    return df_cars_no_nan



def extract_facts(df_cars_facts):
    """
    Extract the facts as required from the cars dataset

    Inputs:
    - df_card_facts: Pandas dataframe of dataset containing the cars dataset

    Returns:
    - Nothing; you can simply us simple print statements that somehow show the result you
      put in the table; the format of the  outputs is not important; see example below.
    """       

    #########################################################################################
    ### Your code starts here ###############################################################



    ### Your code ends here #################################################################
    #########################################################################################








def get_noise_dbscan(X, eps=0.0, min_samples=0):
    
    core_point_indices, noise_point_indices = None, None
    
    #########################################################################################
    ### Your code starts here ###############################################################
    
    ### 2.1 a) Identify the indices of all core points
    
    
    ### Your code ends here #################################################################
    #########################################################################################
    
    
    #########################################################################################
    ### Your code starts here ###############################################################
    
    ### 2.1 b) Identify the indices of all noise points ==> noise_point_indices
    
    
    ### Your code ends here #################################################################
    #########################################################################################
    
    return core_point_indices, noise_point_indices

