import matplotlib.pyplot as plt
import matplotlib.markers as mk
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Read data
    df = pd.read_csv("clean_data.csv", sep='\t')

    year_plot = plt.figure(1)
    # Plot yearbuilt versus price
    plt.scatter(df['YearBuilt'], df['Price'], color='red', marker='.')
    plt.title('YearBuilt versus Price')
    # Don't show the outliers
    plt.xlim((1900, 2018))
    plt.xlabel('YearBuilt', fontsize=14)
    plt.ylabel('Prices', fontsize=14)
    plt.grid(True)


    area_plot = plt.figure(2)
    # Plot buildingarea versus price
    plt.scatter(df['BuildingArea'], df['Price'], color='red', marker='.')
    plt.title('BuildingArea versus Price')
    # Zoom in
    plt.xlim((0, 1000))
    plt.xlabel('BuildingArea', fontsize=14)
    plt.ylabel('Prices', fontsize=14)
    plt.grid(True)


    # Plot distance versus price
    distance_plt = plt.figure(3)
    plt.scatter(df['Distance'], df['Price'], color='red', marker='.')
    plt.title('Distance versus Price')
    plt.xlabel('Distance', fontsize=14)
    plt.ylabel('Prices', fontsize=14)
    plt.grid(True)

    # Plot Car(Parking lots) versus price
    car_plt = plt.figure(4)
    plt.scatter(df['Car'], df['Price'], color='red', marker='.')
    plt.title('# of Parking lots versus Price')
    plt.xlabel('# of Parking lots', fontsize=14)
    plt.ylabel('Prices', fontsize=14)
    plt.grid(True)

    # Show all the plots
    plt.show()