import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import pickle

st.set_page_config(page_title="E-Commerce AI Analyzer", layout="wide")

@st.cache_resource
def load_saved_models():
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

svm_model, tfidf_vectorizer = load_saved_models()

def preprocess_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text)).lower()

def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vec_text = tfidf_vectorizer.transform([cleaned])
    pred = svm_model.predict(vec_text)[0]
    if pred == 1: return 'Positive'
    elif pred == -1: return 'Negative'
    else: return 'Neutral'

st.title("SentiMart: E-Commerce Sentiment Analyzer")
st.write("Powered by Support Vector Machine (SVM) and Natural Language Processing")

st.sidebar.title("Navigation Menu")
module_selection = st.sidebar.radio("Select Module:", ["Live Review Analysis", "Batch CSV Analytics"])

if module_selection == "Live Review Analysis":
    st.subheader("Analyze Single Customer Review")
    review_input = st.text_area("Enter Customer Review Below:", height=150)
    
    if st.button("Analyze Sentiment"):
        if review_input:
            sentiment_result = predict_sentiment(review_input)
            st.subheader("Results")
            if sentiment_result == 'Positive':
                st.success("Positive Sentiment Detected")
                st.info("System Auto-Reply: Thank you for your positive feedback. We are glad you are satisfied with our product.")
            elif sentiment_result == 'Negative':
                st.error("Negative Sentiment Detected")
                st.info("System Auto-Reply: We apologize for the inconvenience. Our support team will review your case shortly.")
            else:
                st.warning("Neutral Sentiment Detected")
                st.info("System Auto-Reply: Thank you for sharing your thoughts. We value your feedback.")
        else:
            st.warning("Please enter a review to analyze.")

elif module_selection == "Batch CSV Analytics":
    st.subheader("Upload CSV for Dashboard Analytics")
    uploaded_csv = st.file_uploader("Upload review dataset (CSV format)", type=["csv"])
    
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.success("File uploaded successfully. Generating visuals...")
        text_column = 'reviews.text' if 'reviews.text' in df.columns else df.columns[0]
        df['Sentiment'] = df[text_column].apply(predict_sentiment)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Sentiment Distribution")
            st.bar_chart(df['Sentiment'].value_counts())
        with col2:
            st.write("Sentiment Proportion")
            s_counts = df['Sentiment'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(s_counts, labels=s_counts.index, autopct='%1.1f%%', startangle=90)
            st.pyplot(fig)
            
        st.write("Most Frequent Words")
        all_words_text = ' '.join(df[text_column].astype(str).tolist())
        wordcloud = WordCloud(width=800, height=300, background_color='white').generate(preprocess_text(all_words_text))
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis('off')
        st.pyplot(fig2)