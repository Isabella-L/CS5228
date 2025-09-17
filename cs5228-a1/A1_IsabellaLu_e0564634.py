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

    
    # clean listing_id
    df_cars_cleaned = df_cars_cleaned[df_cars_cleaned['listing_id'].astype(str).str.isdigit()]
    
    # remove type of vehicle unknown
    df_cars_cleaned = df_cars_cleaned[df_cars_cleaned['type_of_vehicle'] != 'unknown']
    
    # clean invalid manufactured year 
    df_cars_cleaned = df_cars_cleaned[(df_cars_cleaned['manufactured'] < 2025) & (df_cars_cleaned['manufactured'] > 1900)]
   
    # clean invalid no_of_owners
    df_cars_cleaned = df_cars_cleaned[df_cars_cleaned['no_of_owners'] >= 1]
    
    # merge make shortform
    df_cars_cleaned['make'] = df_cars_cleaned['make'].replace('mercedes', 'mercedes-benz')
    df_cars_cleaned['make'] = df_cars_cleaned['make'].replace('rolls', 'rolls-royce')
    
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
    
    # first filter out cancelled listing_id
    df_cars_facts = df_cars_facts[df_cars_facts['listing_id'].astype(str).str.isdigit()]
    print('Number of records after cleaning listing_id: {}'.format(len(df_cars_facts)))

    
    # find make = toyota and model = corolla and manufactured < 2015
    df_toyota_corolla = df_cars_facts[(df_cars_facts['make'] == 'toyota') & (df_cars_facts['model'] == 'corolla') & (df_cars_facts['manufactured'] < 2010)]
    print('1. Number of toyota corolla manufactured before 2010: {}'.format(len(df_toyota_corolla)))

    # top 3 most common make
    print("2. ", df_cars_facts['make'].value_counts().head(3))

    # count of most common model with type_of_vehicle = suv
    df_suv = df_cars_facts[df_cars_facts['type_of_vehicle'] == 'suv']
    print("3. ", df_suv['model'].value_counts().head(1))

    # car's make and price with highest overall price and power < 60
    df_power_60 = df_cars_facts[df_cars_facts['power'] <= 60]
    idx_max_price = df_power_60['price'].idxmax()
    print("4. ", df_power_60.loc[idx_max_price, ['make', 'price']])

    # entry where type_of_vehicle = med size sedan and highest power-to-engine_cap ratio
    midsize = df_cars_facts[df_cars_facts['type_of_vehicle'] =='mid-sized sedan'].copy()
    midsize['power_to_engine'] = round(midsize['power'] / midsize['engine_cap'],2)
    make, model, manufactuered, power_to_engine = midsize.loc[midsize['power_to_engine'].idxmax(),['make','model','manufactured','power_to_engine']]
    print("5. ", make, model, manufactuered, power_to_engine)
    # print(midsize.loc[midsize['power_to_engine'].idxmax()])

    # pearson correlation between price and mileage
    print("6. Correlation between price and mileage: {:.2f}".format(
        df_cars_facts['price'].corr(df_cars_facts['mileage']),
    ))
    print("   Correlation between price and engine_cap: {:.2f}".format(
        df_cars_facts['price'].corr(df_cars_facts['engine_cap'])
    ))

    ### Your code ends here #################################################################
    #########################################################################################








def get_noise_dbscan(X, eps=0.0, min_samples=0):
    
    core_point_indices, noise_point_indices = None, None
    
    #########################################################################################
    ### Your code starts here ###############################################################
    
    n = X.shape[0]
    dist_matrix = euclidean_distances(X, X)
    
    ### 2.1 a) Identify the indices of all core points
    core_point_indices = []
    for i in range(n):
        neighbors = np.where(dist_matrix[i] <= eps)[0]  # indices of neighbors
        if len(neighbors) >= min_samples:
            core_point_indices.append(i)
    ### Your code ends here #################################################################
    #########################################################################################

    #########################################################################################
    ### Your code starts here ###############################################################
    
    ### 2.1 b) Identify the indices of all noise points ==> noise_point_indices
    noise_point_indices = []
    for i in range(n):
        if i in core_point_indices:
            continue  # skip core
        # Check if it has any core point neighbor
        neighbors = np.where(dist_matrix[i] <= eps)[0]
        if not any(j in core_point_indices for j in neighbors):
            noise_point_indices.append(i)
            
    
    ### Your code ends here #################################################################
    #########################################################################################
    
    return core_point_indices, noise_point_indices

