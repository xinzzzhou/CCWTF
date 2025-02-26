import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from config import __root_path__, __raw_data_path__, __save_data__
from transformers import AutoTokenizer, AutoModel
import torch


def load_data(file_path):
    """Load and preprocess the data."""
    data = pd.read_csv(file_path)
    data['summary'] = data['text'].astype(str).apply(lambda x: x.split('Summary of the article:')[1] if 'Summary of the article:' in x else x)
    return data

def calculate_tfidf_representation(data, text_column='text'):
    """Calculate and store TF-IDF representation."""
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data[text_column])
    return tfidf_matrix

def calculate_llama_representation(data, text_column='text', model_id =  "meta-llama/Llama-3.2-1B", batch_size=32):


    import os 

    from huggingface_hub import login

    hugging_face_key = "xxxxx"
    login(hugging_face_key)

  
    """Calculate and store Llama representation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModel.from_pretrained(model_id).to(device)
    
    all_embeddings = []
    
    for i in range(0, len(data), batch_size):
        print(f"Processing batch {i//batch_size+1}/{len(data)//batch_size+1}")

        
        batch_texts = data[text_column][i:i+batch_size].tolist()
        
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        all_embeddings.append(embeddings.cpu().numpy())


    text_embedding_matrix = np.vstack(all_embeddings)
    
    return text_embedding_matrix

def main(raw_data_name='traffic_daily_text_final.csv', method='tfidf', k=10):
    file_path = os.path.join(__raw_data_path__, raw_data_name)
    data = load_data(file_path)
    
    if os.path.exists(os.path.join(__save_data__, f'article_similarity_{method}.npy')):
        print(f"Similarity matrix for {method} already exists.")
        return

    if method == 'llama_1B':
        text_embedding_matrix = calculate_llama_representation(data, model_id="meta-llama/Llama-3.2-1B")
    elif method == 'llama_8B':
        text_embedding_matrix = calculate_llama_representation(data, model_id="meta-llama/Meta-Llama-3.1-8B")
    elif method == 'tfidf':
        text_embedding_matrix = calculate_tfidf_representation(data)
    else:
        raise ValueError("Invalid method. Choose 'tfidf' or 'llama'.")

    similarity_matrix = cosine_similarity(text_embedding_matrix, text_embedding_matrix)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(similarity_matrix)

    # Save the similarity matrix to a npy file
    np.save(os.path.join(__save_data__, f'article_similarity_{method}.npy'), similarity_matrix)
    
    # Sort indices in descending order of similarity for each article
    sorted_indices = np.argsort(-similarity_matrix, axis=1)

    # Remove self-similarity by excluding the first column (which is the article itself)
    sorted_indices = sorted_indices[:, 1:]
    np.save(os.path.join(__save_data__, f'sorted_indices_{method}.npy'), sorted_indices)
    print(sorted_indices)
    print(sorted_indices.shape)

if __name__ == "__main__":
    k = 15
    raw_data_name = 'traffic_daily_text.csv'  
    method = 'llama_8B'
    
    main(raw_data_name, method=method, k=k)
