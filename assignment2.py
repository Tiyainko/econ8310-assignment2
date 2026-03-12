import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3.csv")
test = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3test.csv")

def prepare_data(df):
    df = df.copy()
    dt = pd.to_datetime(df["DateTime"])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["minute"] = dt.dt.minute
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df = df.drop(columns=["id", "DateTime"])
    return df

train_prep = prepare_data(train)
test_prep = prepare_data(test)

X_train = train_prep.drop(columns=["meal"])
y_train = train_prep["meal"].astype(int)

X_test = test_prep.drop(columns=["meal"], errors="ignore")

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

modelFit = model.fit(X_train, y_train)

pred = pd.Series(modelFit.predict(X_test)).astype(int)
