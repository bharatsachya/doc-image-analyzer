from services.embedding_service import generate_embeddings
from utils.similarity import top_n_neighbours

def process_question(prompt, document_text):
    # Embed the document and prompt
    doc_embeddings = generate_embeddings([document_text])
    prompt_embedding = generate_embeddings([prompt])

    # Find similar chunks
    similar_chunks = top_n_neighbours(prompt_embedding, doc_embeddings, document_text)

    # Combine chunks with the base prompt
    full_prompt = "\n".join(similar_chunks) + "\n\n" + prompt

    # Generate response
    response = generate_response(full_prompt)
    return response

def generate_response(prompt, temperature=0.5, max_tokens=200):
    # Call Cohere's generate endpoint (mock implementation)
    return "Generated response based on: " + prompt
