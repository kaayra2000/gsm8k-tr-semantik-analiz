import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE

def get_sentence_embedding(model: AutoModel, tokenizer: AutoTokenizer, text: str) -> np.ndarray:
    """
    Verilen metnin gömüsünü döndüren fonksiyon.

    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        text: Gömüsü alınacak metin

    Returns:
        embedding: Metnin gömüsü
    """
    # metni tokenize et
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Gömüyü oluştur
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embedding = (sum_embeddings / sum_mask).squeeze().numpy()
    
    return embedding

def apply_tsne(embeddings, model_name, perplexity=30, n_iter=3000, random_state=42):
    """
    Gömmelere t-SNE uygulayarak 2 boyuta indirgeme
    
    Args:
        embeddings: Gömme matrisi
        model_name: Model adı
        perplexity: t-SNE perplexity değeri
        n_iter: t-SNE iterasyon sayısı
        random_state: t-SNE random state değeri
    
    Returns:
        embeddings_2d: 2 boyutlu gömme matrisi
    """
    # Embeddings veri tipini ve boyutunu kontrol et
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Gömme boyutu: {embeddings.shape}")
    
    # t-SNE uygula
    print(f"{model_name} için t-SNE uygulanıyor...")
    tsne = TSNE(n_components=2, 
                perplexity=perplexity, 
                n_iter=n_iter,
                init='pca', 
                learning_rate='auto',
                random_state=random_state)
    embeddings_2d = tsne.fit_transform(embeddings)

    return embeddings_2d[:2]