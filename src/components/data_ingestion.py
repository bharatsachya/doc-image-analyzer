import cohere
from typing import Sequence
from config.config import API_KEY
from src.exception import CustomException
import sys
co = cohere.Client(API_KEY)


# Using the Cohere embed endpoint to embed the data into vector data.
def embed(Texts: Sequence[str]):
    try:
        res=co.embed(texts=Texts)
        return res.embeddings
    except Exception as e:
        raise CustomException(e,sys)


# Using the Cohere generate endpoint to return the answer into text data with additional options namely 'temperature' and 'max_tokens'.
def generate(promptt, tmp, maxTokens):
    try:
        res=co.generate(prompt=promptt, temperature=tmp, max_tokens=maxTokens)
        return res
    except Exception as e:
        raise CustomException(e,sys)
