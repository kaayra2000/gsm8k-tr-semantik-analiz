import heapq
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer

from gomme_islemleri import get_token_embeddings

def get_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    İki token matrisi arasındaki benzerliği hesaplar.
    Her token için diğer metindeki en benzer tokeni bulur ve
    bu benzerliklerin ortalamasını döndürür.
    
    Args:
        embedding1: Birinci metin token matris gömmesi
        embedding2: İkinci metin token matris gömmesi
    
    Returns:
        float: Benzerlik skoru (0-1 aralığında)
    """
    # Her iki matrisin ortalama vektörünü al
    emb1_mean = np.mean(embedding1, axis=0, keepdims=True)  # 1 x gömme_boyutu
    emb2_mean = np.mean(embedding2, axis=0, keepdims=True)  # 1 x gömme_boyutu
    
    # Ortalama vektörler arasındaki kosinüs benzerliğini hesapla
    dot_product = np.sum(emb1_mean * emb2_mean)
    norm1 = np.sqrt(np.sum(emb1_mean ** 2))
    norm2 = np.sqrt(np.sum(emb2_mean ** 2))
    
    # Sıfıra bölmeyi önle
    if norm1 * norm2 == 0:
        return 0
        
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)

def find_top5_similar(model: AutoModel, tokenizer: AutoTokenizer, text: str, 
                     target_column: str, dataset: pd.DataFrame) -> List[Tuple[int, float, str, np.ndarray]]:
    """
    Verilen metne en benzeyen 5 kaydı döndüren fonksiyon.

    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        text: Arama yapılacak metin
        target_column: Hedef kolon (hangi kolonda arama yapılacağı)
        dataset: Veri kümesi

    Returns:
        top5_results: En benzer 5 kaydın (indeks, skor, metin, gömme) listesi
    """
    # Text'in token gömmesini al
    text_embedding = get_token_embeddings(model, tokenizer, text)
    
    # Tüm veri kümesini dolaş ve benzerlik skorlarını hesapla
    similarities = []
    for idx, row in dataset.iterrows():
        target_text = row[target_column]
        target_embedding = get_token_embeddings(model, tokenizer, target_text)
        similarity = get_cosine_similarity(text_embedding, target_embedding)
        similarities.append((idx, similarity, target_text, target_embedding))
    
    # Benzerlik skoruna göre sırala (büyükten küçüğe)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # İlk 5 sonucu al
    top5_results = similarities[:5]
    
    # Sonuçları (indeks, skor, metin, gömme) formatında düzenle
    result = [(idx, score, text, embedding) for idx, score, text, embedding in top5_results]
    
    return result