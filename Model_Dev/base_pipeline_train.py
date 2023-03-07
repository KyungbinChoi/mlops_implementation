import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

# 1. get data
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42)

# 2. model development and train
model_pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVR())])
model_pipeline.fit(X_train, y_train)

train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)

train_acc = mean_squared_error(y_true=y_train, y_pred=train_pred, squared=False)
valid_acc = mean_squared_error(y_true=y_valid, y_pred=valid_pred, squared=False)

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)

# 3. save model
joblib.dump(model_pipeline, "model_pipeline.joblib")


