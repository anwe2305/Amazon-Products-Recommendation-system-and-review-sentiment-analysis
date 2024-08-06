import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in text_data:
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(token.lower() for token in nltk.word_tokenize(sentence) if token.lower() not in stop_words))
    return preprocessed_text

# Function to analyze sentiment
def analyze_sentiment(review):
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    for word in blob.words:
        if word.lower() in ["not", "no", "never", "don't", "can't", "won't"]:
            polarity *= -1
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Function to tokenize product name
def tokenize_product_name(name, delimiter='-'):
    tokens = name.split(delimiter)
    return tokens

# Function to create and evaluate the model
def model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    classifier = MultinomialNB()
    classifier.fit(X_train_counts, y_train)
    y_pred = classifier.predict(X_test_counts)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)
    st.write('---------------------------------------------------------------')
    st.write(classification_report(y_test, y_pred))

# Function to recommend products based on category
def recommend_product(df, category):
    category_df = df[df['name'].str.contains(category, case=False)]
    if category_df.empty:
        return "No products found for the specified category."
    product_sentiment = category_df.groupby('name')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
    product_sentiment['overall_sentiment_score'] = product_sentiment['positive'] - product_sentiment['negative']
    product_sentiment_ranked = product_sentiment.sort_values(by='overall_sentiment_score', ascending=False)
    top_positive_products = product_sentiment_ranked.head(5).index.tolist()
    return top_positive_products

def top_recommended_by_rating(df, category):
    # Ensure the necessary columns are present
    if 'name' not in df.columns or 'rating' not in df.columns:
        return "CSV file must contain 'name' and 'rating' columns."
    
    # Filter products by category
    category_df = df[df['name'].str.contains(category, case=False)]
    if category_df.empty:
        return "No products found for the specified category."
    
    avg_ratings = category_df.groupby('name')['rating'].mean()
    
    if avg_ratings.empty:
        return "No products found for the specified category."
    
    # Find the highest rated product
    highest_rated_product = avg_ratings.idxmax()
    highest_avg_rating = avg_ratings.max()
    
    # Get the brand of the highest rated product
    if 'brand' in df.columns:
        recommended_brand = category_df[category_df['name'] == highest_rated_product]['brand'].values[0]
    else:
        recommended_brand = "Brand information not available"
    
    return {
        'recommended_product': highest_rated_product,
        'recommended_brand': recommended_brand,
        'average_rating': highest_avg_rating
    }

# Load data
df = pd.read_csv('AmazonReviews.csv')

# Preprocess the dataset
df.drop(['asin', 'date'], axis=1, inplace=True)
df.dropna(inplace=True)
df['review'] = preprocess_text(df['review'].values)
df['product_name_tokens'] = df['name'].apply(tokenize_product_name)
df['brandName'] = df['name'].str.split('-').str[0]
df['sentiment'] = df['review'].apply(analyze_sentiment)
average_ratings = df.groupby('name')['rating'].mean()
df = df.merge(average_ratings, on='name', suffixes=('', '_avg'))
df.rename(columns={'rating_avg': 'average_rating'}, inplace=True)
df['review_length'] = df['review'].apply(len)
df['word_count'] = df['review'].apply(lambda x: len(x.split()))

# Streamlit app code
st.title("Amazon Reviews Sentiment Analysis and Recommendation System")

# Sentiment analysis for user input
st.write("Sentiment Analysis for Your Review:")
user_review = st.text_input("Enter your review here:")
if user_review:
    sentiment_result = analyze_sentiment(user_review)
    st.write(f"The sentiment of your review is: {sentiment_result}")

# Recommendation system
st.write("Recommendation System:")
input_category = st.text_input("Enter the product category you want:")
if input_category:
    recommendation = recommend_product(df, input_category)
    if isinstance(recommendation, list):
        st.write("Top Recommended Products based on reviews:")
        for i, product in enumerate(recommendation, start=1):
            st.write(f"{i}. {product}")
    else:
        st.write(recommendation)
        message_displayed = True

    result = top_recommended_by_rating(df, input_category)
    if isinstance(result, dict):
        st.write("Top Recommendation based on Ratings:")
        st.write(f"{result['recommended_product']} by {result['recommended_brand']} (Average Rating: {result['average_rating']:.2f})")
    elif isinstance(result, str) and not message_displayed:  # Check if the result is an error message and hasn't been displayed yet
        st.write(result)
        message_displayed = True

# Visualizations
if st.button("Show Visualizations"):
    st.write("Sentiment Distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title('Distribution of Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    st.pyplot(plt)

    st.write("Ratings Distribution:")
    ratings = df["rating"].value_counts()
    numbers = ratings.index
    quantity = ratings.values
    figure = px.pie(df, values=quantity, names=numbers, hole=0.5)
    figure.update_layout(autosize=False, width=500, height=500)
    st.plotly_chart(figure)

    st.write("Word Cloud for Positive Reviews:")
    positive_reviews = ' '.join(df[df['sentiment'] == 'positive']['review'])
    positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
    plt.figure(figsize=(15, 6))
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.title('Word Cloud for Positive Reviews')
    plt.axis('off')
    st.pyplot(plt)

    st.write("Word Cloud for Negative Reviews:")
    negative_reviews = ' '.join(df[df['sentiment'] == 'negative']['review'])
    negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)
    plt.figure(figsize=(15, 6))
    plt.imshow(negative_wordcloud, interpolation='bilinear')
    plt.title('Word Cloud for Negative Reviews')
    plt.axis('off')
    st.pyplot(plt)

    st.write("Sentiment by Product Rating:")
    rating_sentiment_counts = df.groupby('rating')['sentiment'].value_counts().unstack().fillna(0)
    rating_sentiment_counts.plot(kind='bar', stacked=True)
    plt.title('Sentiment by Product Rating')
    plt.xlabel('Product Rating')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment')
    st.pyplot(plt)

# Classification model
# st.write("Classification Model Results:")
# model(df)
