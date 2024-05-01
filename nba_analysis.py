#Load the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Load the data
data = pd.read_csv('NBA_Rookies.csv')

#View the data
data.head()
#Basic information
data.info()
#Describe the data
data.describe()
#Find the duplicates
data.duplicated().sum()
#unique values

# Step 5: Visualize distributions of numerical variables
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.show()

sns.histplot(data['PTSpg'], bins=20, kde=True)
plt.title('Distribution of Points Per Game')
plt.xlabel('Points Per Game')
plt.ylabel('Frequency')
plt.show()

# Step 6: Explore relationships between variables
sns.pairplot(data[['Age', 'PTS', 'TRB', 'AST']])
plt.show()

# Step 7: Analyze categorical variables
sns.countplot(data['Team'])
plt.title('Count of Players by Team')
plt.xticks(rotation=90)
plt.show()

# Additional explorations based on the dataset's specific context

# For example, if there's a target variable:
sns.countplot(data['Target'])
plt.title('Count of Target Variable')
plt.show()

# Or if you want to explore correlations:
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()