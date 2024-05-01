#Load the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Load the data
data = pd.read_csv("NBA_Rookies_InCollegeData.csv")

#View the data
data.head()
#Basic information
data.info()
#Describe the data
data.describe()
#Find the duplicates
data.duplicated().sum()
#unique values

# columns_of_interest = ['yr', 'GP', 'Min_per', 'pts', 'twoPM', 'twoP_per', 'TPM', 'TP_per', 
#                        'FTM', 'FT_per', 'treb', 'ast', 'stl', 'blk']

# # Create histograms for each column
# for column in columns_of_interest:
#     sns.histplot(data[column].dropna(), bins=20, kde=True)
#     plt.title(f'Distribution of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.show()

    