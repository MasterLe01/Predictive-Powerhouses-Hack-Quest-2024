import streamlit as st
import PyPDF2
# import joblib

st.set_page_config(page_title="Project Website Medical report", page_icon=":tada:",layout="wide")
st.subheader("Hi, This is the team Predictive Powerhouses.:wave:")
st.title("Text Summarization of medical report. ")
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Our Goal:")
        st.write("""
            Our goal is to provide Text summarization for medical report using Natural Language Processing(NLP).
            This application helps in providing summarization on text inputs of medical report.
        """)

# Use a pipeline as a high-level helper
# from transformers import pipeline


# pipe = pipeline("summarization", model="Falconsai/medical_summarization")
# joblib.dump(pipe,"medsumm.joblib")


# model=joblib.load("medsumm.joblib")

# def extract_text_from_pdf(file_path):
#     text = ""
#     with open(file_path, "rb") as pdf_file:
#         pdf_reader = PyPDF2.PdfFileReader(pdf_file)
#         for page_num in range(pdf_reader.numPages):
#             page = pdf_reader.getPage(page_num)
#             text += page.extract_text()
#     return text

# st.title("PDF Text Extractor")

# uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if uploaded_file is not None:
#     text = extract_text_from_pdf(uploaded_file)
#     st.text_area("Extracted Text", text, height=400)
#     print(text)

# import streamlit as st
# import PyPDF2
from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit app
st.title("PDF Summarizer")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
    text = ""
    for page_num in range(pdf_reader.getNumPages()):
        page = pdf_reader.getPage(page_num)
        text += page.extract_text()

    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

    st.header("Summary")
    st.write(summary)
    