import os
from io import StringIO
from typing import Sequence

import cohere
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from numpy.linalg import norm
from PIL import Image
import pytesseract
from rake_nltk import Rake
import nltk
import shutil

nltk.download('stopwords')
nltk.download('punkt')



pytesseract.pytesseract.tesseract_cmd = None

# search for tesseract binary in path
@st.cache_resource
def find_tesseract_binary() -> str:
    return shutil.which("tesseract")

# set tesseract binary path
pytesseract.pytesseract.tesseract_cmd = find_tesseract_binary()
print(find_tesseract_binary())
if not pytesseract.pytesseract.tesseract_cmd:
    st.error("Tesseract binary not found in PATH. Please install Tesseract.")

#  loaded local env
load_dotenv()

api=os.getenv('API_KEY')
base=os.getenv('BASE')
# default settings for generation of text
TEMPERATURE = 0.5
MAX_TOKENS = 200
text=""
result=None

co=cohere.Client(api)

def extractTextFromImage(image: Image.Image):
    text = pytesseract.image_to_string(image)
    return text



def highlight_key_points(summary):
    # Use spaCy NER to find important entities (optional)
    rake = Rake()
    rake.extract_keywords_from_text(summary)
    
    # Get the ranked phrases
    ranked_phrases = rake.get_ranked_phrases()
    
    # Highlight the important keywords/phrases
    highlighted_summary = summary
    for phrase in ranked_phrases[:5]:  # Select top 5 phrases
        highlighted_summary = highlighted_summary.replace(phrase, f"**{phrase}**")
    
    return highlighted_summary


# Get the text from pdf file.
def extractTextFromPdf(pdfPath: str):
    text = ""
    with pdfplumber.open(pdfPath) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Creating a dataframe to break information into user defined Chunks.
def processTextInput(text: str, run_id: str = None):
    text = StringIO(text).read()
    CHUNK_SIZE=150
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    df = pd.DataFrame.from_dict({'text': chunks})
    return df

# Converting the dataframe to list of strings.
def convertToList(df):
    df['col']=df[['text']].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    seqOfStrings: Sequence[str]=df['col'].tolist()
    return seqOfStrings

# Using the Cohere embed endpoint to embed the data into vector data.
def embed(Texts: Sequence[str]):
    res=co.embed(texts=Texts, model="small")
    return res.embeddings

# Finding K nearest neighbours to enhance the answer.
def topNNeighbours(promptEmbeddings: np.ndarray, storageEmbeddings: np.ndarray, df, k: int = 5):
	if isinstance(storageEmbeddings, list):
		storageEmbeddings = np.array(storageEmbeddings)
	if isinstance(promptEmbeddings, list):
		storageEmbeddings = np.array(promptEmbeddings)
	similarityMatrix = promptEmbeddings @ storageEmbeddings.T / np.outer(norm(promptEmbeddings, axis=-1), norm(storageEmbeddings, axis=-1))
	numNeighbours = min(similarityMatrix.shape[1], k)
	indices = np.argsort(similarityMatrix, axis=-1)[:, -numNeighbours:]
	listOfStr=df.values.tolist()
	neighbourValues:list=[]
	for idx in indices[0]:
		neighbourValues.append(listOfStr[idx])
	return neighbourValues

# Using the Cohere generate endpoint to return the answer into text data with additional options namely 'temperature' and 'max_tokens'.
def generate(promptt, tmp, maxTokens):
    res=co.generate(prompt=promptt, temperature=tmp, max_tokens=maxTokens)
    return res

# Using the streamlit library to create a user interface for better understanding.
options=st.selectbox("Input type", ["PDF","TEXT","IMAGE"])
embeddings = np.empty((0, 1024))  # Assuming 1024 is the embedding dimension

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
    prompt = st.text_input('Ask a Question')
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
    st.write(result.generations[0].text)
    highlighted_summary = highlight_key_points(result.generations[0].text)
    st.write(highlighted_summary)
    


    