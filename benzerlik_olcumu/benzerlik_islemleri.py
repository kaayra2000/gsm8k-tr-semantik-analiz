import heapq
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer

from gomme_islemleri import get_sentence_embedding

def get_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    İki gömme arasındaki kosinüs benzerliğini hesaplayan fonksiyon.

    Args:
        embedding1: İlk gömme
        embedding2: İkinci gömme

    Returns:
        similarity: Kosinüs benzerliği skoru
    """
    # sklearn'in cosine_similarity fonksiyonu için gömme şeklini yeniden şekillendir
    emb1 = embedding1.reshape(1, -1)
    emb2 = embedding2.reshape(1, -1)
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity

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
    # Min-heap yapısı kullanarak en yüksek 5 benzerliği tut
    top5 = []
    text_embedding = get_sentence_embedding(model, tokenizer, text)
    
    for idx, row in dataset.iterrows():
        target_text = row[target_column]
        target_embedding = get_sentence_embedding(model, tokenizer, target_text)
        similarity = get_cosine_similarity(text_embedding, target_embedding)
        
        # eğer heap boşsa veya 5'ten az eleman varsa ekle
        if len(top5) < 5:
            heapq.heappush(top5, (similarity, idx, target_text, target_embedding))
        # eğer benzerlik en yüksek 5'ten küçükse ekle
        elif similarity > top5[0][0]:
            heapq.heappushpop(top5, (similarity, idx, target_text, target_embedding))
    
    # En yüksek benzerliklerin sırasını tersine çevir
    top5.sort(reverse=True)
    result = [(idx, score, text, embedding) for score, idx, text, embedding in top5]
    
    return result