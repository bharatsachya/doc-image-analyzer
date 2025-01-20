import numpy as np
from numpy.linalg import norm
from src.exception import CustomException
import sys
from src.logger import logging

def topNNeighbours(promptEmbeddings: np.ndarray, storageEmbeddings: np.ndarray, df, k: int = 5):
    if isinstance(storageEmbeddings, list):
        storageEmbeddings = np.array(storageEmbeddings)
    if isinstance(promptEmbeddings, list):
        promptEmbeddings = np.array(promptEmbeddings)
    logging.info("Finding top k neighbours")
    try:
        similarityMatrix = promptEmbeddings @ storageEmbeddings.T / np.outer(norm(promptEmbeddings, axis=-1), norm(storageEmbeddings, axis=-1))
        numNeighbours = min(similarityMatrix.shape[1], k)
        indices = np.argsort(similarityMatrix, axis=-1)[:, -numNeighbours:]
        listOfStr = df.values.tolist()
        neighbourValues: list = []
        for idx in indices[0]:
            neighbourValues.append(listOfStr[idx])
        return neighbourValues
    except Exception as e:
        raise CustomException(e, sys)
