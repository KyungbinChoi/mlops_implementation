import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. reproduce data
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42)

# 2. load model
model_pipeline_load = joblib.load("model_pipeline.joblib")

# 3. validate
load_train_pred = model_pipeline_load.predict(X_train)
load_valid_pred = model_pipeline_load.predict(X_valid)

load_train_acc = mean_squared_error(y_true=y_train, y_pred=load_train_pred, squared=False)
load_valid_acc = mean_squared_error(y_true=y_valid, y_pred=load_valid_pred, squared=False)

print("Load Model Train Accuracy :", load_train_acc)
print("Load Model Valid Accuracy :", load_valid_acc)