import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE

def get_token_embeddings(model: AutoModel, tokenizer: AutoTokenizer, text: str) -> np.ndarray:
    """
    Verilen metnin token gömmelerini matris olarak döndüren fonksiyon.
    Her satır bir tokene, her sütun bir gömme boyutuna karşılık gelir.
    
    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        text: Gömüsü alınacak metin
    
    Returns:
        np.ndarray: Token gömmeleri matrisi (token_sayısı x gömme_boyutu)
    """
    # Metni tokenize et
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Token gömmelerini al
    token_embeddings = outputs.last_hidden_state
    
    # Attention mask kullanarak sadece gerçek token gömmelerini al (padding tokenlarını çıkar)
    attention_mask = inputs['attention_mask'][0]
    valid_tokens = attention_mask.bool()
    
    # Sadece gerçek tokenlerin gömmelerini seç
    token_embs = token_embeddings[0, valid_tokens].detach().cpu().numpy()
    
    return token_embs  # Boyut: (token_sayısı x gömme_boyutu)

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