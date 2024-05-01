import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load your dataset
nba_rookies = pd.read_csv("NBA_Rookies_InCollegeData.csv")

# Preprocessing
X = nba_rookies.drop(['Player', 'Team', 'Year', 'Conf', 'Target'], axis=1)  # Assuming these columns are not used for prediction
y = nba_rookies['Target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Decision Tree classifier
tree = DecisionTreeClassifier(max_depth = 5, random_state=42)

# Train the classifier
tree.fit(X_train, y_train)

# Make predictions
y_pred = tree.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for Decision Tree: {accuracy}")
