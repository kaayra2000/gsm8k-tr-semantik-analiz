import heapq
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer

from gomme_islemleri import get_token_embeddings, apply_tsne

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
                     target_column: str, dataset: pd.DataFrame) -> dict:
    """
    Verilen metne en benzeyen 5 kaydı JSON formatında döndüren fonksiyon.
    Minimum heap kullanarak verimli şekilde en benzer 5 kaydı bulur.

    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        text: Arama yapılacak metin
        target_column: Hedef kolon (hangi kolonda arama yapılacağı)
        dataset: Veri kümesi

    Returns:
        dict: JSON formatında sonuç içeren sözlük
    """
    # Text'in token gömmesini al
    text_embedding = get_token_embeddings(model, tokenizer, text)
    source_text_tsne_embedding = apply_tsne(text_embedding)
    
    # En benzer 5 sonucu tutmak için minimum heap (benzerlik skorunun negatifi ile)
    min_heap = []  # (benzerlik, idx) şeklinde tutacağız
    result_dict = {}  # idx -> veri eşleşmesi için sözlük
    
    # İlk 5 elemanı direkt ekleyelim
    for i, (idx, row) in enumerate(dataset.iterrows()):
        if i >= 5:
            break
            
        target_text = row[target_column]
        target_embedding = get_token_embeddings(model, tokenizer, target_text)
        similarity = get_cosine_similarity(text_embedding, target_embedding)
        target_tsne = apply_tsne(target_embedding)
        
        # Sadece benzerlik ve idx'i heap'e ekle
        heapq.heappush(min_heap, (similarity, idx))
        
        # Diğer bilgileri sözlükte tut
        result_dict[idx] = {
            "text": target_text,
            "score": float(similarity),
            "tsne_embedding": target_tsne.tolist()
        }
    
    # Kalan elemanlar için - eğer en küçükten daha büyükse ekle
    for idx, row in list(dataset.iterrows())[5:]:
        target_text = row[target_column]
        target_embedding = get_token_embeddings(model, tokenizer, target_text)
        similarity = get_cosine_similarity(text_embedding, target_embedding)
        
        # Eğer mevcut en küçük benzerlikten daha büyükse
        if similarity > min_heap[0][0]:
            heapq.heappushpop(min_heap, (similarity, idx))
            
            # Yeni verinin diğer bilgilerini hesapla ve sözlüğe ekle
            target_tsne = apply_tsne(target_embedding)
            result_dict[idx] = {
                "text": target_text,
                "score": float(similarity),
                "tsne_embedding": target_tsne.tolist()
            }
    
    # Min heap'ten en yüksek skorlara göre sırala
    top5_idx_sorted = [idx for _, idx in sorted(min_heap, reverse=True)]
    
    # JSON çıktısını oluştur
    top5_matches = []
    for rank, idx in enumerate(top5_idx_sorted, 1):
        match_data = {}
        match_data["rank"] = rank
        match_data["index"] = idx
        match_data.update(result_dict[idx])
        top5_matches.append(match_data)
    
    # Sonuç JSON'ı oluştur
    result = {
        "source_text": text,
        "source_text_tsne_embedding": source_text_tsne_embedding.tolist(),
        "top5_matches": top5_matches
    }
    
    return result