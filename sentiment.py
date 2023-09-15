import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model
with open('sentiment (1).pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer (make sure it's the same as used during training)
with open('vector.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Sidebar with app logo and info
st.sidebar.image('twit.png', width=200)
st.sidebar.title("Sentiment Analysis App")
st.sidebar.markdown("This app analyzes sentiment in text data and provides sentiment labels and polarity values. It can help you quickly determine the emotional tone and polarity of text, whether it's a review, tweet, or any other form of text.Feel free to enter your own text and see how it's analyzed!")


# Streamlit UI
st.title("Sentiment Analysis Web App")
st.write("Enter text to predict sentiment:")

user_input = st.text_area("Enter text:", "", height=150, max_chars=500)

# Analyze button
if st.button("Analyze", key="analyze_button"):
    if user_input:
        # Transform user input using the pre-fitted TF-IDF vectorizer
        user_input_tfidf = tfidf_vectorizer.transform([user_input])
        
        # Predict the sentiment and get the polarity value
        sentiment = model.predict(user_input_tfidf)[0]  # Ensure you get the first prediction
        
        # Determine the sentiment label and color
        if sentiment > 0.4:
            sentiment_label = "Positive"
            color = "lightgreen"
        elif sentiment < 0:
            sentiment_label = "Negative"
            color = "red"
        elif 0 < sentiment < 0.4 :
            sentiment_label = "Neutral"
            color = "yellow"

        # Display sentiment and polarity with color
        st.subheader("Sentiment Analysis Result:")
        st.markdown(f"<span style='color:{color}; font-size:30px; font-weight:bold;'>{sentiment_label}</span>", unsafe_allow_html=True)
        st.markdown(f"**Polarity Value:** {sentiment:.3f}", unsafe_allow_html=True)  # Increased text size and rounded polarity to 3 decimal places

    else:
        st.warning("Please enter some text for analysis.")
