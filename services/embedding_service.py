import cohere
from config.config import API_KEY

co = cohere.Client(API_KEY)

def generate_embeddings(texts, model="small"):
    response = co.embed(texts=texts, model=model)
    return response.embeddings
