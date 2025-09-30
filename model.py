import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import re
path = r"C:\Users\{username}\Desktop\mbti_1.csv"

# Read CSV
df = pd.read_csv(path, header=None, dtype=str, engine="python")

# The first column is personalities. Value counts of unique personalities
personality_counts = df[0].value_counts()

#print(personality_counts)

# Bar chart
personality_counts.plot(kind="bar", figsize=(10,6))

plt.title("Personality Type Frequency")
plt.xlabel("Personality Type")
plt.ylabel("Count")
plt.show(block=False)

df.columns = ["personality", "text"]

#cleaning text data

df["text"] = df["text"].str.strip(" '") #remove quotation
print("hello world init")
# some other examples of cleaning text data w/ explanations for more complex cleaning = may enhance model accuracy


df['text'] = df['text'].apply(lambda x: re.sub(r'http\S+|www\S+|@\w+|#\w+|\d+|[^\w\s]', '', x))
df['text'] = df['text'].str.strip().str.replace(r'\s+', ' ', regex=True)
df = df[df['text'].str.len() > 0]


# df["text"] = df["text"].str.replace(r'http\S+|www.\S+', '', case=False) # remove URLs
# d["text"] = df["text"].str.replace(r'@\w+', '', case=False) # remove mentions
# d["text"] = df["text"].str.replace(r'#\w+', '', case=False) # remove hashtags
# d["text"] = df["text"].str.replace(r'\d+', '', case=False) # remove numbers
# d["text"] = df["text"].str.replace(r'[^\w\s]', '', case=False) # remove punctuation
# d["text"] = df["text"].str.lower() # convert to lowercase
# d["text"] = df["text"].str.strip() # remove leading/trailing whitespace
# d["text"] = df["text"].str.replace(r'\s+', ' ', case=False) # replace multiple spaces with single space
# d["text"] = d["text"].replace('', np.nan) # replace empty strings with NaN
# d = d.dropna(subset=["text"]) # drop rows with NaN in 'text' column


#Splitting data into train/test and featrues etc
x = df["text"] # features
y = df["personality"] # labels (personalities)



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
#stratify=y to maintain the same proportion of classes in train and test sets as in the original dataset, however since we only have one y, or label which is the personality type
#stratify will give error. There needs to be at least 2 for it to work.

#control log
print("hello world 1 ")

# vectorizing data
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
x_train_vectors = vectorizer.fit_transform(x_train)
x_test_vectors = vectorizer.transform(x_test)

print("hello world 2 ")


# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train_vectors, y_train)

print("hello world 3")

#Evaluation
y_pred=model.predict(x_test_vectors)
print("Accuracy:", accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))
print("hello world 4 ")


# Predicting personality type for new text input
def predict_personality(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

print("hello world 5 ")

# Example usage
new_text = "I love making friends and partying."
predicted_personality = predict_personality(new_text)
print(f"Predicted Personality Type: {predicted_personality}")