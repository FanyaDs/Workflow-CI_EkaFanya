import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def main(data_path):
    print(f"📂 Load dataset: {data_path}")
    df = pd.read_csv(data_path)
    print("Jumlah data:", len(df))

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']

    print("Shape TF-IDF:", X.shape)

    # Kalau data besar, pakai stratify; kalau kecil, skip stratify
    if len(df) >= 30:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("Akurasi:", acc)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        mlflow.log_metric("accuracy", acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="namadataset_preprocessing.csv")
    parser.add_argument("-f", "--file", help="ignore jupyter arg", default=None)
    args = parser.parse_args()
    main(args.data)
