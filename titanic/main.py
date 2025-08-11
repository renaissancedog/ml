import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Path of the file to read
file_path = 'titanic/train.csv'
train_data = pd.read_csv(file_path)
test_data = pd.read_csv('titanic/test.csv')
combine = [train_data, test_data]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



# Separate target and features
y = train_data["Survived"]
print(train_data.columns)
# Select features and apply get_dummies
features = ["Pclass","Fare","Sex", 'Age','IsAlone']
X = pd.get_dummies(train_data[features])
val_X = pd.get_dummies(test_data[features])

# Align columns of val_X to X
val_X = val_X.reindex(columns=X.columns, fill_value=0)
print(train_data.describe(include=['O']))
# Train the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# Make predictions
predictions = model.predict(val_X)

# Save predictions to a CSV file
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('titanic/submission.csv', index=False)
print("Your submission was successfully saved!")

# Read and describe the output
r = pd.read_csv("titanic/submission.csv")
print(r.describe())
print(r['Survived'].value_counts()[1]/418)
