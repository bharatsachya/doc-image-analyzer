import cohere
from config.config import API_KEY
import sys
from src.exception import CustomException
from src.logger import logging

co = cohere.Client(API_KEY)

def generate_embeddings(texts, model="small"):
    logging.info("Generating embeddings")
    try:
        embeddings = co.embed(texts, model=model)
        return embeddings
    except Exception as e:
        raise CustomException(e,sys)
