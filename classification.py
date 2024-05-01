import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
nba_rookies = pd.read_csv("NBA_Rookies_InCollegeData.csv")
college_rookies = pd.read_csv("NBA_makers_LastYearOnly_CollegeBasketballData.csv")

# Preprocessing
nba_rookies.drop(['Player', 'Team', 'Year', 'Conf', 'Age', 'TOV', 'PF', 'TOVpg', 'PFpg'], axis=1, inplace=True)
print(nba_rookies.head())
# Split data into features (X) and target (y)
X = nba_rookies.drop(['Target'], axis=1)
y = nba_rookies['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_pred = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

print("KNN Accuracy:", knn_accuracy)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_accuracy)

# ---------- Define Decision Tree classifiers with different depths ----------
small_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
big_tree = DecisionTreeClassifier(max_depth=8, random_state=42)

# Split data for underfitting and overfitting analysis
small_data_train, _, small_data_train_y, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=42)
big_data_train, big_data_train_y = X_train, y_train

print("=================================================")
print("Results demonstrating underfitting and overfitting")
print("=================================================")

# Function to evaluate and print accuracy
def evaluate_accuracy(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

# Evaluate decision trees
print('\nSmall Decision Tree (trained on small data):')
evaluate_accuracy(small_tree, small_data_train, small_data_train_y, X_test, y_test)

print('\nBig Decision Tree (trained on small data):')
evaluate_accuracy(big_tree, small_data_train, small_data_train_y, X_test, y_test)

print('\nSmall Decision Tree (trained on big data):')
evaluate_accuracy(small_tree, big_data_train, big_data_train_y, X_test, y_test)

print('\nBig Decision Tree (trained on big data):')
evaluate_accuracy(big_tree, big_data_train, big_data_train_y, X_test, y_test)
