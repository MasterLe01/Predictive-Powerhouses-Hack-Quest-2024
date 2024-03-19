
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import chardet
from sklearn.metrics import classification_report



def load_data():
    df_train = pd.read_csv("train.csv", encoding=chardet.detect(open("train.csv", "rb").read())["encoding"])
    df_train.dropna(axis=0, inplace=True)
    df_test = pd.read_csv("test.csv", encoding=chardet.detect(open("test.csv", "rb").read())["encoding"])
    df_test.dropna(axis=0, inplace=True)
    return df_train, df_test

def load_model():
    model = joblib.load("sentiment.joblib")
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df_train["text"])
    return model, vectorizer

# Set Streamlit page configuration
st.set_page_config(page_title="Project Website", page_icon=":tada:", layout="wide")

# Header section
st.subheader("Hi, This is the team Predictive Powerhouses.👋")
st.title("Sentiment Analysis")

df_train, df_test = load_data()
model, vectorizer = load_model()

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Our Goal:")
        st.write("""
            Our goal is to provide sentiment analysis using machine learning techniques.
            This application helps in predicting sentiment based on text inputs.
        """)
        

with st.container():
    st.write("---")
    st.header("Machine Learning Model")
    st.write("This section allows you to input text and get sentiment predictions.")

    # Text input
    new_text = st.text_input("Enter your text:", "I am not happy")

    # Button to trigger prediction
    if st.button("Predict Sentiment"):
        new_features = vectorizer.transform([new_text])
        new_predictions = model.predict(new_features)
        st.write("Sentiment Prediction:", new_predictions[0])

with st.container():
    # with st.expander:
        st.write("---")
        st.header("Model Evaluation")
        st.write("This section displays the classification report for the test data.")

        # Display classification report
        test_features = vectorizer.transform(df_test["text"])
        test_predictions = model.predict(test_features)
        report = classification_report(df_test["sentiment"], test_predictions)
        st.write("Classification Report:")
        st.text_area("Report:", report)
