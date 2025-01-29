import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Example dataset: [height, weight, shoe size, gender]
'''
data = [
    [170, 70, 42, "Male"],
    [160, 55, 37, "Female"],
    [180, 80, 44, "Male"],
    [165, 60, 38, "Female"],
    [175, 75, 43, "Male"],
    [155, 50, 36, "Female"],
    [185, 85, 45, "Male"],
    [150, 45, 35, "Female"]
]'''

# Read csv file
data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')

# Convert dataset into a DataFrame
df = pd.DataFrame(data, columns=["Height", "Weight", "BMI Index", "Gender"])

# Prepare features (X) and target (y)
X = df[["Height", "Weight", "BMI Index"]]
y = df["Gender"]

# Encode the target variable
y = y.map({"Male": 1, "Female": 0})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
'''
print('X_train : ')
print(X_train.head())
print(X_train.shape)

print('')
print('X_test : ')
print(X_test.head())
print(X_test.shape)
 
print('')
print('y_train : ')
print(y_train.head())
print(y_train.shape)
 
print('')
print('y_test : ')
print(y_test.head())
print(y_test.shape)
'''
# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)

# Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_preds = dtc.predict(X_test)

# Support Vector Machine (SVM)
svm = SVC()
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)

# k-Nearest Neighbors (k-NN)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)

# Evaluate each model
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_reg_preds))
print("Decision Tree Accuracy:", accuracy_score(y_test, dtc_preds))
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))
print("k-NN Accuracy:", accuracy_score(y_test, knn_preds))

# Classification Report for Logistic Regression 
print("\nClassification Report (Logistic Regression):\n")
print(classification_report(y_test, log_reg_preds))

# Classification Report for Decision Tree Classifier
print("\nClassification Report (Decision Tree Classifier):\n")
print(classification_report(y_test, dtc_preds))
