import numpy as np
from typing import List, Dict
from dosya_islemleri import get_calculated_probabilities_size, append_probability

def get_cosine_similarity(embedding1, embedding2) -> float:
    """
    İki gömme vektörü arasındaki kosinüs benzerliğini hesaplar.
    
    Args:
        embedding1: İlk gömme vektörü (liste veya numpy dizisi)
        embedding2: İkinci gömme vektörü (liste veya numpy dizisi)
        
    Returns:
        float: İki vektör arasındaki kosinüs benzerliği (-1 ile 1 arasında)
    """
    
    # Gömme vektörlerini numpy dizilerine dönüştür
    if not isinstance(embedding1, np.ndarray):
        embedding1 = np.array(embedding1)
    if not isinstance(embedding2, np.ndarray):
        embedding2 = np.array(embedding2)
    
    # Vektörlerin boyutlarını kontrol et
    if embedding1.shape != embedding2.shape:
        raise ValueError("Gömme vektörlerinin boyutları eşleşmiyor!")
    
    # Sıfır vektörlerine karşı korunma
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0  # Sıfır vektörleri arasında anlamlı bir benzerlik hesaplanamaz
    
    # Kosinüs benzerliği hesaplama formülü: cos(θ) = (A·B) / (||A|| * ||B||)
    # Nokta çarpımı
    dot_product = np.dot(embedding1, embedding2)
    
    # Normların çarpımı
    norms_product = norm1 * norm2
    
    # Kosinüs benzerliği
    similarity = dot_product / norms_product
    
    return similarity


def calculate_and_save_similarity_scores(embeddings: List[Dict], save_prefix: str):
    """
    Verilen gömme vektörlerinin benzerlik skorlarını hesaplar.
    
    Args:
        embeddings: Gömme vektörlerini içeren liste (sözlükler)
        save_prefix: Kaydedilecek dosya adının öneki
    """
    # Toplam hesaplanacak benzerlik sayısı
    embed_len = len(embeddings)
    total_process_count = embed_len * embed_len
    
    # Şimdiye kadar hesaplanmış benzerlik sayısını al
    start_index = get_calculated_probabilities_size(save_prefix)
    
    # Genel ilerlemeyi göster
    print(f"Hali hazırda hesaplanmış benzerlik skorları sayısı: {start_index} / {total_process_count}, bu kayıtlar atlanıyor.")
    print("\n")
    
    # Devam edilecek i ve j indekslerini hesapla
    i_start = start_index // embed_len
    j_start = start_index % embed_len
    
    # İşlemde kaldığımız yerden devam edelim
    for i in range(i_start, embed_len):
        upper_element = embeddings[i]
        
        # İlk satırda j_start'tan başla, diğer satırlarda baştan başla
        start_j = j_start if i == i_start else 0
        
        for j in range(start_j, embed_len):
            lower_element = embeddings[j]
            
            # Soru ve cevap gömme vektörlerini al
            question_embedding = upper_element["question_embedding"]
            answer_embedding = lower_element["answer_embedding"]
            
            # Benzerlik skorunu hesapla
            similarity_score = get_cosine_similarity(question_embedding, answer_embedding)
            
            # Skoru dosyaya ekle
            append_probability(save_prefix, upper_element["index"], lower_element["index"], similarity_score)
            
            # İlerlemeyi göster (tek satırda güncellenen)
            print(f"{i+1}/{embed_len} ve {j+1}/{embed_len} benzerlik skoru hesaplandı ve kaydedildi.", end="\r")