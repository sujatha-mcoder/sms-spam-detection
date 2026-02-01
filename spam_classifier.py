import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(
    "sms_spam.csv",
    sep="\t",
    header=None,
    names=["Label", "Message"]
)

# Convert labels to numbers
df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

# Features and label
X = df["Message"]
y = df["Label"]

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = MultinomialNB()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Test new message
new_sms = ["Congratulations! You won a free gift card"]
new_sms_vec = vectorizer.transform(new_sms)

result = model.predict(new_sms_vec)

if result[0] == 1:
    print("Message is SPAM")
else:
    print("Message is NOT SPAM")
