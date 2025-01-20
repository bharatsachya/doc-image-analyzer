import pandas as pd
from io import StringIO
import pandas as pd
from io import StringIO
from typing import Sequence
import sys
from src.exception import CustomException
from src.logger import logging

# Converting the dataframe to list of strings.
def convertToList(df):
    logging.info("Converting dataframe to list of strings")
    df['col']=df[['text']].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    seqOfStrings: Sequence[str]=df['col'].tolist()
    return seqOfStrings


def processTextInput(text: str, run_id: str = None):
    logging.info("Processing text input")
    try:
        text = StringIO(text).read()
        CHUNK_SIZE = 150
        chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        df = pd.DataFrame.from_dict({'text': chunks})
        return df
    except Exception as e:
        raise CustomException(e, sys)

