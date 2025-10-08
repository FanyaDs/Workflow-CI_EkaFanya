import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

df = pd.read_csv("namadataset_preprocessing.csv")
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_enc = X_train.astype(str)
X_test_enc = X_test.astype(str)

mlflow.set_tracking_uri("file:///content/mlruns")
mlflow.set_experiment("Sentimen_RandomForest_EkaFanya")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_enc.values.reshape(-1, 1), y_train)
    y_pred = model.predict(X_test_enc.values.reshape(-1, 1))
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1)
    mlflow.sklearn.log_model(model, "model")

print("✅ Model berhasil dilatih dan dicatat di MLflow")
