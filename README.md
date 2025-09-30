// This project is a Machine Learning model designed to predict MBTI personality types based on text data. It utilizes the MBTI Dataset from Kaggle( https://www.kaggle.com/datasets/datasnaek/mbti-type/data )

// Dataset
The dataset is a CSV file containing entries in the format:
[MBTI], ["text data"]
The text data was scraped from various internet sources.

Note: The dataset is highly unbalanced, which may affect model performance.

//Features
Counts the number of posts per MBTI type.
Performs basic data cleaning on text entries.
Trains a Logistic Regression model for personality prediction.

//Model Performance
The current model provides basic predictions.
Accuracy is limited due to dataset imbalance.

//Performance can be improved by:
Collecting more balanced data
Incorporating additional textual features

//Usage
from predictor import predict_personality
example:
text = "I enjoy structured plans and prefer deep conversations over casual small talk."
predicted_personality = predict_personality(text)
print(f"Predicted MBTI Type: {predicted_personality}")

//Future Improvements
Implement data augmentation to balance MBTI classes.
Experiment with different ML algorithms.
