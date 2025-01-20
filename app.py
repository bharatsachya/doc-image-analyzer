import numpy as np
import pandas as pd

import streamlit as st
from PIL import Image
import pytesseract
import nltk
import shutil

from config.config import TEMPERATURE
from config.config import MAX_TOKENS
from config.config import base

from src.components.data_extraction import extractTextFromImage, extractTextFromPdf
from src.components.data_transformation import processTextInput, convertToList
from src.components.data_model import topNNeighbours
from src.components.feature_extraction import highlight_key_points
from src.components.data_ingestion import embed, generate

nltk.download('stopwords')
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab',quiet=True)

pytesseract.pytesseract.tesseract_cmd = None
result = None
# search for tesseract binary in path
@st.cache_resource
def find_tesseract_binary() -> str:
    return shutil.which("tesseract")

# set tesseract binary path
pytesseract.pytesseract.tesseract_cmd = find_tesseract_binary()
print(find_tesseract_binary())

if not pytesseract.pytesseract.tesseract_cmd:
    st.error("Tesseract binary not found in PATH. Please install Tesseract.")

# Using the streamlit library to create a user interface for better understanding.
options=st.selectbox("Input type", ["PDF","TEXT","IMAGE"])
embeddings = np.empty((0, 1024))  # Assuming 1024 is the embedding dimension
text=""
if options=="PDF":
    pdfFile=st.file_uploader("Drag and drop or upload your PDF file", type=["pdf"])
    if pdfFile is not None:
        text=extractTextFromPdf(pdfFile)
    if text is not None:
        df=processTextInput(text)
elif options == "TEXT":
    text = st.text_area("Paste the Document")
    if text is not None:
        df = processTextInput(text)
elif options == "IMAGE":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"]) 
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        text = extractTextFromImage(image)
    if text is not None:   
        df = processTextInput(text)

summary_length = st.selectbox("Select Summary Length", ["Short", "Medium", "Long"])

if text!="":
    listOfText=convertToList(df)
    embeddings=embed(listOfText)

if df is not None:
    prompt = st.text_input('Ask a Question:', placeholder = 'summarize document or text from image')
    advancedOpt = st.checkbox('Advanced Options')
    if advancedOpt:
        TEMPERATURE = st.slider('Temperature', min_value=0.0, max_value=1.0, value=TEMPERATURE)
    if summary_length == "Short":
        MAX_TOKENS = 100  # Approx. 50-100 words
    elif summary_length == "Medium":
        MAX_TOKENS = 250  # Approx. 150-250 words
    else:
        MAX_TOKENS = 500  # Approx. 300-500 words


if df is not None and prompt != "":
    basePrompt = base
    promptEmbeddings = embed([prompt])
    augPrompts = topNNeighbours(np.array(promptEmbeddings), embeddings, df)
    joinedPrompt = '\n'.join(str(neighbour) for neighbour in augPrompts) + '\n\n' + basePrompt + '\n' + prompt + '\n'
    result = generate(joinedPrompt, TEMPERATURE, MAX_TOKENS)
    # After generating the summary


if result is not None:
    highlighted_summary = highlight_key_points(result.generations[0].text)
    st.write(highlighted_summary)
    


    