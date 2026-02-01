import argparse
import os
import io
import zipfile
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

def download_ucismsspam():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    print("Downloading SMS Spam Collection from UCI...")
    resp = urllib.request.urlopen(url)
    z = zipfile.ZipFile(io.BytesIO(resp.read()))
    with z.open("SMSSpamCollection") as f:
        lines = [l.decode("utf-8", errors="ignore").strip() for l in f.readlines()]
    rows = [l.split("\t", 1) for l in lines if "\t" in l]
    df = pd.DataFrame(rows, columns=["label", "text"])
    return df

def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8", engine="python", error_bad_lines=False)
    # Adapt to common column names
    if set(["label","text"]).issubset(df.columns):
        return df[["label","text"]].dropna()
    if set(["v1","v2"]).issubset(df.columns):
        return df.rename(columns={"v1":"label","v2":"text"})[["label","text"]].dropna()
    # fallback: try first two columns
    cols = df.columns.tolist()
    return df[[cols[0], cols[1]]].rename(columns={cols[0]:"label", cols[1]:"text"}).dropna()

def train_and_save(df, model_path):
    df['label'] = df['label'].str.lower().map(lambda x: 1 if x.strip() == "spam" else 0)
    X = df['text'].astype(str)
    y = df['label'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_df=0.9)),
        ("nb", MultinomialNB())
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("Classification report:\n", classification_report(y_test, preds, target_names=["ham","spam"]))
    joblib.dump(pipeline, model_path)
    print("Saved model to", model_path)
    return pipeline

def predict_single(pipeline, text):
    p = pipeline.predict([text])[0]
    return "spam" if p == 1 else "ham"

def main():
    parser = argparse.ArgumentParser(description="Train or use a simple spam detector")
    parser.add_argument("--data", "-d", help="Path to CSV with columns (label,text). If omitted downloads UCI SMS Spam Collection.")
    parser.add_argument("--model", "-m", default="spam_model.joblib", help="Path to save/load model")
    parser.add_argument("--predict", "-p", help="If provided, load model and predict the given message string")
    args = parser.parse_args()

    if args.predict:
        if not os.path.exists(args.model):
            print("Model", args.model, "not found. Train first or provide --data to train.")
            return
        pipeline = joblib.load(args.model)
        print("Prediction:", predict_single(pipeline, args.predict))
        return

    if args.data:
        if not os.path.exists(args.data):
            print("Data file not found:", args.data)
            return
        df = load_csv(args.data)
    else:
        df = download_ucismsspam()

    train_and_save(df, args.model)

if __name__ == "__main__":
    main()