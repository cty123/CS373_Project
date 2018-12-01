import pandas as pd
import datetime
import numpy as np
from sklearn.utils import shuffle
import pandas as pd

# This function read the original raw data into the dataframe, 
# and then clean missing entries and saves our interested features
# in "./clean_data.csv"
def gen_clean_data():
    df = pd.read_csv("Melbourne_housing_FULL.csv")
    df = df[['Price', 'Distance', 'Car', 'BuildingArea', 'YearBuilt']]
    clean_df = df.dropna(how='any', axis=0)
    clean_df.to_csv("clean_data.csv", sep='\t')
    return clean_df

def get_data_points():
    df = pd.read_csv("clean_data.csv", sep='\t')
    df = df.dropna(how='any', axis=0)
    df = shuffle(df)
    x_points = df[['Distance', 'Car','BuildingArea', 'YearBuilt']]
    y_points = df[['Price']]
    
    return np.array(x_points), np.array(y_points)

if __name__ == "__main__":
    gen_clean_data().head(10)