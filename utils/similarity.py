import numpy as np
from numpy.linalg import norm

def top_n_neighbours(prompt_embedding, storage_embeddings, document_text, k=5):
    if len(storage_embeddings) == 0:
        return []    
    if(isinstance(prompt_embedding, list)):
        prompt_embedding = np.array(prompt_embedding)
    if(isinstance(storage_embeddings, list)):
        storage_embeddings = np.array(storage_embeddings)

    similarity = prompt_embedding @ storage_embeddings.T / (
        norm(prompt_embedding, axis=-1) * norm(storage_embeddings, axis=-1)
    )
    top_indices = np.argsort(similarity, axis=-1)[:, -k:]
    return [document_text[i] for i in top_indices.flatten()]
