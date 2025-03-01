# Import necessary libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import os
from wordcloud import WordCloud

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Set random seed for reproducibility
np.random.seed(42)

# Function to preprocess text
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Stemming
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    # Join words back into a single string
    return ' '.join(words)

# Function to load and preprocess the IMDB dataset
def load_imdb_data():
    try:
        # Try to load from sklearn datasets
        from sklearn.datasets import load_files
        reviews = load_files('aclImdb', shuffle=False)
        X, y = reviews.data, reviews.target
        # Convert bytes to strings
        X = [x.decode('utf-8') for x in X]
    except:
        # If not available, download from keras
        from tensorflow.keras.datasets import imdb
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
        
        # Get the word index
        word_index = imdb.get_word_index()
        # Reverse word index to get words
        reverse_word_index = {value: key for key, value in word_index.items()}
        
        # Convert indices to words
        X_train_text = [' '.join([reverse_word_index.get(i - 3, '?') for i in sequence]) for sequence in X_train]
        X_test_text = [' '.join([reverse_word_index.get(i - 3, '?') for i in sequence]) for sequence in X_test]
        
        X = X_train_text + X_test_text
        y = np.concatenate([y_train, y_test])
    
    # Create DataFrame
    df = pd.DataFrame({'review': X, 'sentiment': y})
    # Map sentiment values to meaningful labels (0=negative, 1=positive)
    df['sentiment'] = df['sentiment'].map({0: 'negative', 1: 'positive'})
    
    return df

# Function to train the model
def train_model(df):
    # Create a models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Check if model already exists
    if os.path.exists('models/imdb_model.pkl') and os.path.exists('models/vectorizer.pkl'):
        # Load the model and vectorizer
        with open('models/imdb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        # Preprocess the reviews
        df['processed_review'] = df['review'].apply(preprocess_text)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_review'], df['sentiment'], test_size=0.2, random_state=42
        )
        
        # Vectorize the text
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train a logistic regression model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Save the model and vectorizer
        with open('models/imdb_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
            
        # Print evaluation metrics
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
    
    return model, vectorizer

# Function to predict sentiment
def predict_sentiment(review, model, vectorizer):
    # Preprocess the review
    processed_review = preprocess_text(review)
    # Vectorize the review
    review_vec = vectorizer.transform([processed_review])
    # Predict sentiment
    prediction = model.predict(review_vec)[0]
    # Get prediction probability
    proba = model.predict_proba(review_vec)[0]
    return prediction, proba

# Streamlit App
def main():
    st.title("IMDB Movie Review Sentiment Analysis")
    
    # Sidebar
    st.sidebar.header("Options")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Analyze Your Review", "Dataset Insights"])
    
    # Load data and train model
    with st.spinner("Loading IMDB dataset and training model..."):
        df = load_imdb_data()
        model, vectorizer = train_model(df)
    
    if page == "Home":
        st.header("Welcome to the IMDB Sentiment Analysis App")
        st.write("""
        This application analyzes movie reviews and determines whether they express a positive or negative sentiment.
        The model has been trained on the IMDB dataset, a popular benchmark for sentiment analysis.
        
        Use the sidebar to navigate to different sections of the app:
        - **Analyze Your Review**: Input your own movie review and see the sentiment prediction
        - **Dataset Insights**: Explore the IMDB dataset and model performance
        """)
        
        # Display sample reviews
        st.subheader("Sample Reviews from the Dataset")
        st.write("Here are a few examples of reviews and their sentiments:")
        
        # Sample 3 positive and 3 negative reviews
        sample_pos = df[df['sentiment'] == 'positive'].sample(3)
        sample_neg = df[df['sentiment'] == 'negative'].sample(3)
        samples = pd.concat([sample_pos, sample_neg])
        
        for idx, row in samples.iterrows():
            with st.expander(f"{row['sentiment'].title()} Review Example"):
                st.write(row['review'][:500] + "..." if len(row['review']) > 500 else row['review'])
    
    elif page == "Analyze Your Review":
        st.header("Analyze Your Movie Review")
        st.write("Enter your movie review below and our model will predict its sentiment.")
        
        # Text input for user review
        user_review = st.text_area("Your movie review:", height=200)
        
        if st.button("Analyze"):
            if user_review.strip() == "":
                st.error("Please enter a review to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    # Predict sentiment
                    prediction, proba = predict_sentiment(user_review, model, vectorizer)
                    
                    # Display result
                    st.subheader("Prediction Result")
                    
                    # Determine confidence percentage based on prediction
                    confidence = proba[1] if prediction == 'positive' else proba[0]
                    confidence_percent = round(confidence * 100, 2)
                    
                    # Create columns for result display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Sentiment", prediction.title())
                    
                    with col2:
                        st.metric("Confidence", f"{confidence_percent}%")
                    
                    # Display probability bar
                    st.subheader("Sentiment Probability")
                    prob_df = pd.DataFrame({
                        'Sentiment': ['Negative', 'Positive'],
                        'Probability': [proba[0], proba[1]]
                    })
                    
                    fig, ax = plt.subplots(figsize=(10, 2))
                    sns.barplot(x='Probability', y='Sentiment', data=prob_df, 
                                palette=['#ff9999', '#99ff99'], orient='h', ax=ax)
                    ax.set_xlim(0, 1)
                    ax.set_title("Sentiment Probability")
                    st.pyplot(fig)
    
    elif page == "Dataset Insights":
        st.header("IMDB Dataset Insights")
        
        # Basic dataset information
        st.subheader("Dataset Overview")
        st.write(f"Total number of reviews: {len(df)}")
        
        # Count sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        
        # Create columns for sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Sentiment Distribution:")
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                   colors=['#ff9999', '#99ff99'], startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        
        with col2:
            st.write("Sentiment Counts:")
            st.bar_chart(sentiment_counts)
        
        # Process a sample of reviews to show word clouds
        st.subheader("Most Common Words by Sentiment")
        
        # Take a sample for faster processing
        sample_df = df.sample(min(1000, len(df)))
        
        # Process reviews by sentiment
        positive_reviews = ' '.join(sample_df[sample_df['sentiment'] == 'positive']['review'].apply(preprocess_text))
        negative_reviews = ' '.join(sample_df[sample_df['sentiment'] == 'negative']['review'].apply(preprocess_text))
        
        # Create columns for word clouds
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Positive Reviews")
            if positive_reviews:
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                     colormap='Greens', max_words=100).generate(positive_reviews)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        
        with col2:
            st.write("Negative Reviews")
            if negative_reviews:
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                     colormap='Reds', max_words=100).generate(negative_reviews)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

if __name__ == "__main__":
    main()