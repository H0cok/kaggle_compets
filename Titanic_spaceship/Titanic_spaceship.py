import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.neighbors._base
import sys

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

survived = train_data.loc[train_data.Transported == True]["Transported"]
print(f"% of survived: {len(survived) / len(train_data)}")

HomePlanets = train_data["HomePlanet"].unique()[:-1]
for planet in HomePlanets:
    df = train_data.loc[train_data.HomePlanet == planet]
    df_transported = df.loc[df.Transported == True]
    print(f"{planet}: percent of survived:{len(df_transported) / len(df)}")

CryoSleep = train_data.loc[train_data.CryoSleep == True]
CryoSleep_survive = CryoSleep.loc[CryoSleep.Transported == True]
print(len(CryoSleep))
print(f"CryoSleep percent of survived:{len(CryoSleep_survive) / len(CryoSleep)}")

for service in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
    df = train_data.loc[train_data[service] > 1000]
    df_s = df.loc[df.Transported == True]
    print(f"{service}: {len(df_s) / len(df)}")

Age = train_data.loc[train_data.Age < 18]
Age_s = Age.loc[Age.Transported == True]
print(f"Age percent of survived:{len(Age_s) / len(Age)}")

Vip = train_data.loc[train_data.VIP == True]
Vip_s = Vip.loc[Vip["Transported"] == True]
print(f"% of survived: {len(Vip_s) / len(Vip)}")

DesPlanets = train_data["Destination"].unique()[:-1]
for planet in DesPlanets:
    df = train_data.loc[train_data.Destination == planet]
    df_transported = df.loc[df.Transported == True]
    print(f"{planet}: percent of survived:{len(df_transported) / len(df)}")


train_data.dropna(subset=["Cabin"], inplace=True)

train_data.loc[train_data["Cabin"].str[-1] == "S", "side"] = False
train_data.loc[train_data["Cabin"].str[-1] == "P", "side"] = True
train_data["port"] = train_data["Cabin"].str[0]
ports = train_data["port"].unique()
# for port in ports:
#         df = train_data.loc[train_data.port == port]
#         df_transported = df.loc[df.Transported == True]
#         print(f"{port}: percent of survived:{len(df_transported)/len(df)}")
print(train_data.isnull().sum())
Vip = train_data.loc[train_data.side == False]
Vip_s = Vip.loc[Vip["Transported"] == True]
print(f"% of survived: {len(Vip_s) / len(Vip)}")

train_data["VIP"].fillna(0, inplace=True)
test_data["VIP"].fillna(0, inplace=True)


print(train_data)

test_data.loc[test_data["Cabin"].str[-1] == "S", "side"] = False
test_data.loc[test_data["Cabin"].str[-1] == "P", "side"] = True

test_data["port"] = test_data["Cabin"].str[0]
for feature in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
    print(f"empty {feature}: {len(train_data.loc[train_data[feature] != 0])}")





y = train_data["Transported"]
features = ["port", "HomePlanet", "CryoSleep", "Age", "RoomService", "FoodCourt",
            "ShoppingMall", "Spa", "VRDeck", "side", "VIP", "Destination"]


X_test = pd.get_dummies(test_data[features])
X = pd.get_dummies(train_data[features])

print(X.isnull().sum())
print(X_test.isnull().sum())

imputer = MissForest(criterion='squared_error',n_estimators=400)
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed)
X_imputed.columns = X.columns


X_test_imputed = imputer.fit_transform(X_test)
X_test_imputed = pd.DataFrame(X_test_imputed)
X_test_imputed.columns = X_test.columns







print(X_imputed)
print(X_test_imputed)

model = RandomForestClassifier(random_state=1, n_estimators=20000, max_depth=13)
model.fit(X_imputed, y)






predictions = model.predict(X_test_imputed)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Transported': predictions})
output.to_csv('submission.csv', index=False)
print("Submission was successfully saved!")
