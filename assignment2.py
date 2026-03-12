import pandas as pd
from xgboost import XGBClassifier

train = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3.csv")
test = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3test.csv")

def prepare_data(df):
    df = df.copy()
    dt = pd.to_datetime(df["DateTime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dayofweek"] = dt.dt.dayofweek
    df["dayofyear"] = dt.dt.dayofyear
    df["quarter"] = dt.dt.quarter
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_morning"] = ((dt.dt.hour >= 5) & (dt.dt.hour < 11)).astype(int)
    df["is_lunch"] = ((dt.dt.hour >= 11) & (dt.dt.hour < 15)).astype(int)
    df["is_afternoon"] = ((dt.dt.hour >= 15) & (dt.dt.hour < 17)).astype(int)
    df["is_dinner"] = ((dt.dt.hour >= 17) & (dt.dt.hour < 22)).astype(int)
    df["is_late"] = ((dt.dt.hour >= 22) | (dt.dt.hour < 5)).astype(int)
    df = df.drop(columns=["DateTime", "id"], errors="ignore")
    return pd.get_dummies(df, drop_first=False)

train_prep = prepare_data(train)
test_prep = prepare_data(test)

X_train = train_prep.drop(columns=["meal"])
y_train = train_prep["meal"].astype(int)

X_test = test_prep.drop(columns=["meal"], errors="ignore")
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

modelFit = model.fit(X_train, y_train)

pred = pd.Series((modelFit.predict_proba(X_test)[:, 1] >= 0.5).astype(int))
