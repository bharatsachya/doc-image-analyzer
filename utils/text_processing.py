from io import StringIO
import pandas as pd

def chunk_text(text, chunk_size=150):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def text_to_dataframe(text, chunk_size=150):
    chunks = chunk_text(text, chunk_size)
    return pd.DataFrame({'text': chunks})
