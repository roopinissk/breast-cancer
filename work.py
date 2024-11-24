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

#KNN
X= df_normalized.drop(columns=['diagnosis'])
y= df_normalized['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

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

X= df_normalized.drop(columns=['diagnosis'])
y= df_normalized['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

y_pred= decision_tree.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)