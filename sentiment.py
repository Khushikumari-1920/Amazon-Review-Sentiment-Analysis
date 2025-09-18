import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("Amazon Reviews Sentiment Analysis")

# Step 1: Load the Dataset
df = pd.read_csv('amazon.csv', on_bad_lines='skip', delimiter=',', encoding='utf-8')

st.write("Columns in dataset:", df.columns)

# Step 2: Data Preprocessing
df.dropna(subset=['review_content'], inplace=True)

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
st.subheader("Sentiment Distribution")

fig, ax = plt.subplots()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
st.pyplot(fig)

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
st.subheader("Aspect Sentiment Distribution")
st.write(aspect_summary.head())

# Visualize Aspect Sentiment
aspect_counts = aspect_summary.melt(var_name='Aspect', value_name='Sentiment')

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.countplot(data=aspect_counts, x='Sentiment', hue='Aspect', ax=ax2)
ax2.set_xlabel("Sentiment")
ax2.set_ylabel("Count")
st.pyplot(fig2)

