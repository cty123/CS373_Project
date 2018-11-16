import pandas as pd
import datetime

def gen_clean_data():
    df = pd.read_csv("Melbourne_housing_FULL.csv")
    df = df[['Price', 'Distance', 'Car', 'BuildingArea', 'YearBuilt']]
    clean_df = df.dropna(how='any', axis=0)
    clean_df.to_csv("clean_data.csv", sep='\t')
    return clean_df

print("Here's the data preview")
print("==========================")
print(gen_clean_data().head(10))

