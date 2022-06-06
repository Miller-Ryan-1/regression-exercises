# wrangle
from acquire import get_zillow_data
import pandas as pd
import numpy as np
from scipy import stats
import sklearn.preprocessing

import warnings
warnings.filterwarnings("ignore")

def wrangle_zillow():
    '''
    Acquires and prepares zillow data for exploration and modeling.
    Function Actions: Pulls Data -> Drops Nulls -> Converts datatypes to int (where possible) -> eliminates odd values
    '''
    # Pull data using an acquire function
    df = get_zillow_data()

    # Drop all nulls from dataset
    df = df.dropna()

    # Convert to integers where we can
    df = df.astype({'bedroomcnt':'int', 'calculatedfinishedsquarefeet':'int', 'taxvaluedollarcnt':'int', 'yearbuilt':'int','fips':'int'})

    # Rename annoying long names
    df = df.rename(columns={'calculatedfinishedsquarefeet':'sqft','taxvaluedollarcnt':'value'})

    # Eliminate the funky values
    df = df[df['sqft'] > 400]
    df = df[df['sqft'] < 10000]
    df = df[df['value'] > 10000]
    df = df[df['value'] < 20000000]
    df = df[df['taxamount'] > 200]
    df = df[df['taxamount'] < 300000]
    df = df[df['bathroomcnt'] > 0]
    df = df[df['bedroomcnt'] > 0]
    df = df[df['bathroomcnt'] < 8]
    df = df[df['bedroomcnt'] < 8]

    # Convert Fips to Names
    df['fips_name'] = np.where(df.fips == 6037, 'Los Angeles', np.where(df.fips == 6059, 'Orange','Ventura') )
    df = df.drop(columns = 'fips')

    return df

def scale_zillow(df_train,df_validate,df_test):
    # Create the object, dropping categorical and target variables
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(df_train.drop(columns=['fips_name','value']))

    # Fit the data
    df_train_scaled = pd.DataFrame(scaler.transform(df_train.drop(columns=['fips_name','value'])),columns=df_train.drop(columns=['fips_name','value']).columns.values).set_index([df_train.index.values])
    df_validate_scaled = pd.DataFrame(scaler.transform(df_validate.drop(columns=['fips_name','value'])),columns=df_validate.drop(columns=['fips_name','value']).columns.values).set_index([df_validate.index.values])
    df_test_scaled = pd.DataFrame(scaler.transform(df_test.drop(columns=['fips_name','value'])),columns=df_test.drop(columns=['fips_name','value']).columns.values).set_index([df_test.index.values])

    # Add back in the fips
    df_train_scaled['fips_name'] = df_train['fips_name']
    df_validate_scaled['fips_name'] = df_validate['fips_name']
    df_test_scaled['fips_name'] = df_test['fips_name']

    # Add back in the target
    df_train_scaled['value'] = df_train['value']
    df_validate_scaled['value'] = df_validate['value']
    df_test_scaled['value'] = df_test['value']

    # Encode fips_name
    dummy_df_train = pd.get_dummies(df_train_scaled[['fips_name']], dummy_na=False, drop_first=False)
    dummy_df_validate = pd.get_dummies(df_validate_scaled[['fips_name']], dummy_na=False, drop_first=False)
    dummy_df_test = pd.get_dummies(df_test_scaled[['fips_name']], dummy_na=False, drop_first=False)
    
    df_train_scaled = pd.concat([df_train_scaled, dummy_df_train], axis=1)
    df_validate_scaled = pd.concat([df_validate_scaled, dummy_df_validate], axis=1)
    df_test_scaled = pd.concat([df_test_scaled, dummy_df_test], axis=1)

    return df_train_scaled, df_validate_scaled, df_test_scaled

