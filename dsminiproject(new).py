# -*- coding: utf-8 -*-
"""DSMiniProject(NEW).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13e1zFQe6JcsB6wwPzUicRF__Vz13sgER

# ***Data Science Mini Project***

#**Topic:-Amazon Product reviews Sentiment Analysis**

Group Members:-
Sakshi Minde(8018)
Anwesha Damle(8019)

Class:- TE AI&DS

Batch:- T1

**Importing Libraries**
"""

import pandas as pd     #import all the necessary libraries for the text analysis
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import r
from textblob import TextBlob # it provides the dictionary for proprecessing textual data
import plotly.express as px

# Download NLTK resources
nltk.download('vader_lexicon')   #download the required packages
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))   #initializing the english stop words- common words

from sklearn.model_selection import train_test_split   #libraries for the classification model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

"""**Import the training dataset**"""

df = pd.read_csv('AmazonReviews.csv')   #Importing the dataset into the pandas dataframe
df.head(10)

df.shape  #dimensions of the dataset

"""## **Exploratery data analysis (EDA)**"""

df.describe()  #statistical description of the data

df.info()   #features and data type description

prod = df['name'].unique()    #unique product names
prod

"""##**Data Preprocessing**"""

col= 'asin'    #dropping unecessary column for the analysis
df.drop(col, axis=1, inplace=True)

col1= 'date'
df.drop(col1, axis=1, inplace=True)

df.isnull().sum()  #checking for the null values in the dataset

#drops row will null values
df=df.dropna()

stopwords = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text_data):
    preprocessed_text = []

    for sentence in tqdm(text_data):
        # Removing punctuations
        sentence = re.sub(r'[^\w\s]', '', sentence)

        # Converting lowercase and removing stopwords
        preprocessed_text.append(' '.join(token.lower() for token in nltk.word_tokenize(sentence) if token.lower() not in stopwords))

    return preprocessed_text

preprocessed_review = preprocess_text(df['review'].values)   #printing the preproccesed reviews
df['review'] = preprocessed_review
df

"""## **Dataset Formatting**"""

def tokenize_product_name(name, delimiter='-'):   #function to retrun tokens of product name
    tokens = name.split(delimiter)
    return tokens

# Apply tokenization to each product name and save to a new column
df['product_name_tokens'] = df['name'].apply(tokenize_product_name)

# Print the updated DataFrame
df

df['brandName'] = df['name'].str.split('-').str[0]   #creating seperate column for brand name
df.head()

df['brandName'].value_counts()   #unique brand names count

"""# **Sentiment Analysis**"""

# Define a function to analyze sentiment
def analyze_sentiment(review):
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity

    for word in blob.words:
        if word.lower() in ["not", "no", "never", "don't", "can't", "won't"]:
            polarity *= -1  # Reverse the polarity if a negation is found

    if polarity > 0:    #checking polarity for positive negative
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'


# Apply sentiment analysis to each review
df['sentiment'] = df['review'].apply(analyze_sentiment)

df['sentiment']

sentiment_counts = df['sentiment'].value_counts()   #count of number of pos neg reviews
sentiment_counts

df   #display of the positive negative classification of the reviews in sentiment column

average_ratings = df.groupby('name')['rating'].mean()   #adding an average review product wise column

# Merge the average ratings back into the original DataFrame
df = df.merge(average_ratings, on='name', suffixes=('', '_avg'))

# Rename the new column to 'average_rating'
df.rename(columns={'rating_avg': 'average_rating'}, inplace=True)

# Print the updated DataFrame
df

# Feature Engineering
# Extract additional features from the text data
df['review_length'] = df['review'].apply(len)  # Length of the review
df['word_count'] = df['review'].apply(lambda x: len(x.split()))  # Number of words in the review
df

"""## **Data Visualization**"""

sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])   #representation of no. of pos neg neutral reviews
plt.title('Distribution of Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

ratings = df["rating"].value_counts()   #pie chart based on ratings
numbers = ratings.index
quantity = ratings.values

figure = px.pie(df, values=quantity, names=numbers,hole = 0.5)
figure.update_layout(autosize=False, width=500, height=500)
figure.show()

# Create word clouds for positive and negative reviews
from wordcloud import WordCloud
positive_reviews = ' '.join(df[df['sentiment'] == 'positive']['review'])
negative_reviews = ' '.join(df[df['sentiment'] == 'negative']['review'])

positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)

plt.figure(figsize=(15, 6))    #positive review word cloud
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Positive Reviews')
plt.axis('off')

plt.figure(figsize=(15, 6))    #negative review word cloud
plt.subplot(1, 2, 1)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Negative Reviews')
plt.axis('off')

# Compare sentiment across product categories
rating_sentiment_counts = df.groupby('rating')['sentiment'].value_counts().unstack().fillna(0)
rating_sentiment_counts.plot(kind='bar', stacked=True)
plt.title('Sentiment by Product Rating')
plt.xlabel('Product Rating')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.show()

"""## **Classification Model Creation**"""

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# Predict sentiment on the test set
y_pred = classifier.predict(X_test_counts)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print('---------------------------------------------------------------')
# Evaluation of  the model using various values
print(classification_report(y_test, y_pred))

"""## **Top 10 product recommendations based on sentiment analysis**"""

# Aggregate sentiment scores for each product
product_sentiment = df.groupby('name')['sentiment'].value_counts(normalize=True).unstack().fillna(0)

# Calculate overall sentiment score for each product (you can customize the aggregation method as needed)
product_sentiment['overall_sentiment_score'] = product_sentiment['positive'] - product_sentiment['negative']

# Rank products by sentiment score
product_sentiment_ranked = product_sentiment.sort_values(by='overall_sentiment_score', ascending=False)

# Filter and recommend top-ranked products with positive sentiments
top_positive_products = product_sentiment_ranked.head(10).index.tolist()

# Print the top 10 recommended products with positive sentiments
print("Top 10 Recommended Products with Positive Sentiments:")
print(top_positive_products)

"""## **Recommendation system of products based on input using the ratings**"""

def recommend_product(category):
    # Filter dataset for the specified category
    category_df = df[df['name'].str.contains(category, case=False)]

    if category_df.empty:
        return "No products found for the specified category."

    # Calculate average rating for each product
    avg_ratings = category_df.groupby(['brandName', 'name']).agg({'rating': 'mean'}).reset_index()

    # Find product with the highest average rating
    highest_rated_product = avg_ratings.loc[avg_ratings['rating'].idxmax()]

    # Extract brand name and product name
    recommended_brand = highest_rated_product['brandName']
    recommended_product = highest_rated_product['name']

    return f"Recommended product: {recommended_product} by {recommended_brand} (Average Rating: {highest_rated_product['rating']})"

# Example usage:
input_category = input("Enter the product you want:- ")
recommendation = recommend_product(input_category)
print(recommendation)

















































































