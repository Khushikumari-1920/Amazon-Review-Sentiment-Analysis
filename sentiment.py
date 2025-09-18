import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Dataset
df = pd.read_csv('amazon.csv', on_bad_lines='skip', delimiter=',', encoding='utf-8')

# Print column names for debugging (optional)
print("Columns in dataset:", df.columns)

# Step 2: Data Preprocessing
print(df.info())  # Check for null values and structure
if 'review_content' not in df.columns:
    raise KeyError("Column 'review_content' not found in the dataset. Check column names.")

df.dropna(subset=['review_content'], inplace=True)  # Drop null reviews

# Function to clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['cleaned_review'] = df['review_content'].apply(clean_text)

# Step 3: Sentiment Analysis
def analyze_sentiment(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['cleaned_review'].apply(analyze_sentiment)

# Step 4: Count and Visualize Sentiments
sentiment_counts = df['Sentiment'].value_counts()

plt.figure(figsize=(8, 4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Step 5: Aspect-Based Sentiment Analysis
aspect_keywords = {
    'Quality': ['quality', 'good', 'bad', 'excellent', 'poor'],
    'Price': ['price', 'cost', 'expensive', 'cheap'],
    'Delivery': ['delivery', 'shipping', 'arrived', 'late']
}

def aspect_sentiment(review, aspect_keywords):
    aspect_sentiments = {aspect: 'Not Mentioned' for aspect in aspect_keywords.keys()}
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            if keyword in review:
                aspect_sentiments[aspect] = analyze_sentiment(review)
                break
    return aspect_sentiments

df['Aspect Sentiment'] = df['cleaned_review'].apply(lambda x: aspect_sentiment(x, aspect_keywords))

# Step 6: Analyze Aspect Sentiment
aspect_summary = pd.DataFrame(df['Aspect Sentiment'].tolist())
print(aspect_summary.head())

# Visualize Aspect Sentiment
aspect_counts = aspect_summary.melt(var_name='Aspect', value_name='Sentiment')

plt.figure(figsize=(10, 5))
sns.countplot(data=aspect_counts, x='Sentiment', hue='Aspect')
plt.title('Aspect Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.legend(title='Aspect', loc='upper right')
plt.show()
