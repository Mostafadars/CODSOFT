#* Import the needed modules*#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#* Load the dataset *#
data = pd.read_csv("data/tested.csv")


#* Data preprocessing *#
# Remove unnecessary columns
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Fill missing age values with median
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing embarked values with mode
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Fill the missing fare values with median
data['Fare'].fillna(data['Fare'].median(), inplace=True)


#* Encode categorical variables *#
label_encoder = LabelEncoder()

data['Sex'] = label_encoder.fit_transform(data['Sex'])

data['Embarked'] = label_encoder.fit_transform(data['Embarked'])


#* Split the data into features (X) and target (y) *#
feature = data.drop('Survived', axis=1)

target = data['Survived']


#* Split data into training and testing sets *#
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)


#* Train a Random Forest Classifier *#
clf = RandomForestClassifier(random_state=42)

clf.fit(X_train, y_train)


#* Make predictions on the test set *#
y_pred = clf.predict(X_test)


#* Evaluate the model *#
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')


#* Feature importances *#
importances = clf.feature_importances_

features = feature.columns

plt.bar(features, importances)
plt.xticks(rotation=45)
plt.show()