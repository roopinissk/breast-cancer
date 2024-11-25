import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

#Load the csv data
df = pd.read_csv('breast_cancer.csv')
df.head()

df.shape

print(df['diagnosis'].unique() )
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'diagnosis'.
df['diagnosis']= label_encoder.fit_transform(df['diagnosis'])
df['diagnosis'].unique()

#Remove nan column unnamed32
df = df.drop(columns=["Unnamed: 32"])

#Check if there are any Nan
has_Nan = df.isna().any().any()  # Returns True if any NaN values are found

print("Any NaN values in the DataFrame:", has_Nan)

scaler = MinMaxScaler()

normalized_data = scaler.fit_transform(df.iloc[:, 2:])  # Exclude 'id' and 'diagnosis' columns from normalization
df_normalized = pd.DataFrame(normalized_data, columns=df.columns[2:])
df_normalized['diagnosis'] = df['diagnosis']

##############################################

#Split the data into training and testing sets
X= df_normalized.drop(columns=['diagnosis'])
y= df_normalized['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

#KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", confusion_matrix)

classification = classification_report(y_test, y_pred)
print("Classification Report:", classification)

#########################################

#Decision tree classifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

y_pred= decision_tree.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#trying to improve accuracy
from sklearn.model_selection import GridSearchCV

#setting up prameter grid
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(decision_tree, param_grid, cv=5)
grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
print("Best Parameters:", best_params)

#updating the decision tree

best_params = {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10}

decision_tree = DecisionTreeClassifier(**best_params, random_state= 50)
decision_tree.fit(X_train, y_train)

y_pred= decision_tree.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#cross validating because im getting the same score

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(decision_tree, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy Mean:", cv_scores.mean())
print("Cross-Validation Accuracy Std Dev:", cv_scores.std())