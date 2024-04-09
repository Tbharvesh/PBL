import os
import pandas as pd
import streamlit as st
from dotenv.main import load_dotenv
import google.generativeai as genai
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@st.cache_resource 
def load_gemini_model():
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.0-pro")
    return model

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

nltk.download('punkt')  # Ensuring necessary NLTK data is downloaded.

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    stems = [stemmer.stem(word) for word in lemmas]
    return ' '.join(stems)

def get_relevant_data(question, df, model):
    vectorizer = TfidfVectorizer(stop_words='english')
    processed_question = preprocess_text(question)
    df['processed_input'] = df['input'].apply(preprocess_text)
    
    # Vectorize the processed inputs and the question.
    vectors = vectorizer.fit_transform(df['processed_input'].tolist() + [processed_question])
    similarity = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    
    # Identify the most similar input based on cosine similarity.
    most_similar_index = similarity.argmax()
    highest_similarity_score = similarity.max()  # Get the highest similarity score.
    
    # Define a threshold for similarity to consider the input as relevant.
    relevance_threshold = 0.4  # Adjust this threshold based on your needs
    if highest_similarity_score > relevance_threshold:
        relevant_input = df.iloc[most_similar_index]['input']
        prompt = (
    "You are an AI chatbot trained specifically to provide support for mental health issues. "
    "Firstly, if the question seems like a general conversation starter or not directly related to mental health support, "
    "like a simple 'Hi' or 'How can I help you?', respond in a welcoming and helpful manner. "
    "For more detailed inquiries related to mental health, use the following input from the dataset to provide accurate, "
    "understandable, and empathetic support: '{}' ".format(relevant_input) +
    "Based on this context, here's the question you need to respond to: '{}' ".format(question) +
    "If the question does not align with the kind of support you offer, please guide the user to ask a question more relevant to mental health support."
)
        generated_content = model.generate_content(prompt)


        return generated_content.text
    else:
        return "The question does not seem to be closely related to the dataset's content."



st.title("Question Answering with Your CSV Data")
st.write("Upload your CSV file and ask questions based on its content.")

uploaded_file = st.file_uploader("Choose a CSV file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    
    question = st.text_input("Ask a question about the data:")
    if question:
        model = load_gemini_model()  # Load the model
        try:
            relevant_output = get_relevant_data(question, df, model)
            st.write(relevant_output)
        except Exception as e:
            st.error(f"Error generating answer: {e}")