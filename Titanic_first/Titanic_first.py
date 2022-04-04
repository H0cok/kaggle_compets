import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
train_data.dropna(subset="Age", inplace=True)
test_data["Age"].fillna(test_data["Age"].mean(), inplace=True)





y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Submission was successfully saved!")