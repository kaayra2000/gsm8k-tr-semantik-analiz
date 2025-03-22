import numpy as np
from typing import List, Dict
from dosya_islemleri import get_calculated_probabilities_size, append_probability, append_top1_top5_results_json, get_calculated_top5_results_size
import heapq
import torch
def get_cosine_similarity(embedding1, embedding2, device='auto') -> float:
    """
    İki gömme vektörü arasındaki kosinüs benzerliğini hesaplar.
    
    Args:
        embedding1: İlk gömme vektörü (liste veya numpy dizisi)
        embedding2: İkinci gömme vektörü (liste veya numpy dizisi)
        device: Hesaplama için kullanılacak cihaz ('auto', 'cpu', 'cuda')
               'auto' değeri mevcutsa GPU kullanacaktır
        
    Returns:
        float: İki vektör arasındaki kosinüs benzerliği (-1 ile 1 arasında)
    """
    
    # GPU kullanılabilirlik kontrolü
    use_gpu = False
    if device == 'auto' or device == 'cuda':
        try:
            if torch.cuda.is_available():
                use_gpu = True
            elif device == 'cuda':
                print("GPU istenildi ancak kullanılamıyor. CPU kullanılacak.")
        except ImportError:
            if device == 'cuda':
                print("PyTorch yüklü değil. GPU kullanılamıyor. CPU kullanılacak.")
    
    if use_gpu:
        # Torch tensorlarına dönüştür
        if isinstance(embedding1, np.ndarray):
            embedding1 = torch.from_numpy(embedding1).float().cuda()
        elif isinstance(embedding1, list):
            embedding1 = torch.tensor(embedding1, dtype=torch.float32).cuda()
        else:
            embedding1 = embedding1.cuda()  # Zaten torch tensor ise
            
        if isinstance(embedding2, np.ndarray):
            embedding2 = torch.from_numpy(embedding2).float().cuda()
        elif isinstance(embedding2, list):
            embedding2 = torch.tensor(embedding2, dtype=torch.float32).cuda()
        else:
            embedding2 = embedding2.cuda()  # Zaten torch tensor ise
        
        # Vektörlerin boyutlarını kontrol et
        if embedding1.shape != embedding2.shape:
            raise ValueError("Gömme vektörlerinin boyutları eşleşmiyor!")
        
        # Sıfır vektörlerine karşı korunma
        norm1 = torch.linalg.norm(embedding1)
        norm2 = torch.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0  # Sıfır vektörleri arasında anlamlı bir benzerlik hesaplanamaz
        
        # Kosinüs benzerliği hesaplama
        dot_product = torch.dot(embedding1, embedding2)
        norms_product = norm1 * norm2
        similarity = dot_product / norms_product
        
        return similarity.item()  # GPU'dan CPU'ya taşı ve Python float'a dönüştür
        
    else:
        # CPU ile hesaplama - orijinal numpy kodu
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
        dot_product = np.dot(embedding1, embedding2)
        norms_product = norm1 * norm2
        similarity = dot_product / norms_product
        
        return similarity

def eliminate_element_embeddings(element: Dict) -> Dict:
    """
    Verilen bir elemandan gömme vektörlerini kaldırır.
    
    Args:
        element: Gömme vektörlerini içeren eleman (sözlük)
        
    Returns:
        Dict: Gömme vektörleri olmayan eleman
    """
    # question_embedding ve answer_embedding anahtarlarını kaldır
    element_copy = element.copy()
    element_copy.pop("question_embedding", None)
    element_copy.pop("answer_embedding", None)
    return element_copy
    

def calculate_and_save_similarity_scores(embeddings: List[Dict], save_prefix: str, device='auto'):
    """
    Verilen gömme vektörlerinin benzerlik skorlarını hesaplar.
    
    Args:
        embeddings: Gömme vektörlerini içeren liste (sözlükler)
        save_prefix: Kaydedilecek dosya adının öneki
        device: Hesaplama için kullanılacak cihaz ('auto', 'cpu', 'cuda')
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
            eliminated_upper_element = eliminate_element_embeddings(upper_element)
            eliminated_lower_element = eliminate_element_embeddings(lower_element)
            
            # Benzerlik skorunu hesapla
            similarity_score = get_cosine_similarity(question_embedding, answer_embedding, device)
            
            # Skoru dosyaya ekle
            append_probability(save_prefix, eliminated_upper_element, eliminated_lower_element, similarity_score)
            
            # İlerlemeyi göster (tek satırda güncellenen)
            print(f"{i+1}/{embed_len} ve {j+1}/{embed_len} benzerlik skoru hesaplandı ve kaydedildi.", end="\r")

def get_single_top5_matches(data, source_index, is_question_to_answer):
    """
    Belirli bir kaynak için en yüksek 5 benzer eşleşmeyi döndürür.
    Bu sürüm, probability değerini float'a çevirir ve tuple'a id(item) ekleyerek
    sözlük karşılaştırma hatasını önler.
    """
    # Veri filtreleme: "source" ya da "target" alanı hem dictionary hem de int olabilir.
    if is_question_to_answer:
        filtered_data = [item for item in data if (item.get("source") if isinstance(item.get("source"), int) 
                                                     else item.get("source", {}).get("index")) == source_index]
        target_key = "target"
        source_item = filtered_data[0]["source"] if isinstance(filtered_data[0]["source"], dict) else {"index": filtered_data[0]["source"], "question": ""}
    else:
        filtered_data = [item for item in data if (item.get("target") if isinstance(item.get("target"), int) 
                                                     else item.get("target", {}).get("index")) == source_index]
        target_key = "source"
        source_item = filtered_data[0]["target"] if isinstance(filtered_data[0]["target"], dict) else {"index": filtered_data[0]["target"], "answer": ""}
    
    # Eğer eşleşme yoksa None döndür.
    if not filtered_data:
        return None

    source_text = source_item.get("question") if is_question_to_answer else source_item.get("answer", "")
    
    # Min-heap kullanarak en yüksek 5 olasılığı izleyelim.
    # Her tuple'a (probability, id(item), item) ekleyerek, aynı probability durumunda
    # benzersiz id değeri sayesinde karşılaştırma yapılır.
    top5_heap = []
    for item in filtered_data:
        try:
            probability = float(item["probability"])
        except (ValueError, TypeError):
            probability = 0.0  # Çevrilemezse varsayılan olarak 0.0 kullan
        
        if len(top5_heap) < 5:
            heapq.heappush(top5_heap, (probability, id(item), item))
        elif probability > top5_heap[0][0]:
            heapq.heappushpop(top5_heap, (probability, id(item), item))
    
    # Heap'i olasılığa göre azalan sırada sırala
    top5 = [t[2] for t in sorted(top5_heap, key=lambda x: x[0], reverse=True)]
    
    if top5:
        if isinstance(top5[0].get(target_key), dict):
            first_index = top5[0][target_key].get("index")
        else:
            first_index = top5[0].get(target_key)
        first_probability = float(top5[0]["probability"])
    else:
        first_index = None
        first_probability = None

    # Top5 indekslerini çıkar
    top5_indexs = []
    for item in top5:
        if isinstance(item.get(target_key), dict):
            top5_indexs.append(item[target_key].get("index"))
        else:
            top5_indexs.append(item.get(target_key))
    
    # Top5 sonuçlarını oluştur
    top5_results = [
        {
            "rank": i,
            "index": item[target_key].get("index") if isinstance(item.get(target_key), dict) else item.get(target_key),
            "probability": float(item["probability"]),
            "dest_text": item[target_key].get("answer") if (is_question_to_answer and isinstance(item.get(target_key), dict)) 
                         else (item[target_key] if isinstance(item.get(target_key), str) else "")
        } for i, item in enumerate(top5)
    ]
    
    result = {
        "source_index": source_index,
        "source_text": source_text,
        "source_dest_type": "question_to_answer" if is_question_to_answer else "answer_to_question",
        "1st_index": first_index,
        "1st_probability": first_probability,
        "top5_indexs": top5_indexs,
        "top5_results": top5_results
    }
    
    return result


def get_and_save_all_top5_matches(data, save_prefix: str):
    """
    Veri kümesindeki tüm öğeleri analiz eder ve sonuçları döndürür.
    
    Parametreler:
        data: Analiz edilecek veri kümesi
        save_prefix: Kaydedilecek dosya adının öneki
    """
    # Benzersiz kaynak indekslerini al
    unique_source_indices = set()
    calculated_question_size = get_calculated_top5_results_size(save_prefix, is_question_to_answer=True)
    calculated_answer_size = get_calculated_top5_results_size(save_prefix, is_question_to_answer=False)
    print(f"{calculated_question_size} soru→cevap ve {calculated_answer_size} cevap→soru benzerlik sonuçları hali hazırda hesaplanmış.")
    for item in data:
        unique_source_indices.add(item["source"]["index"])
    unique_source_indices_list = sorted(list(unique_source_indices))
    for idx in range(calculated_question_size, len(unique_source_indices_list)):
        source_index = unique_source_indices_list[idx]
        # Soru→cevap analizi
        q_to_a_result = get_single_top5_matches(data, source_index, True)
        if q_to_a_result:
            append_top1_top5_results_json(save_prefix, True, q_to_a_result)
        print(f"{idx+1}/{len(unique_source_indices_list)} benzerlik sonuçları hesaplandı ve kaydedildi.", end="\r")
    for idx in range(calculated_answer_size, len(unique_source_indices_list)):
        source_index = unique_source_indices_list[idx]
        # Cevap→soru analizi
        a_to_q_result = get_single_top5_matches(data, source_index, False)
        if a_to_q_result:
            append_top1_top5_results_json(save_prefix, False, a_to_q_result)
        print(f"{idx+1}/{len(unique_source_indices_list)} benzerlik sonuçları hesaplandı ve kaydedildi.", end="\r")
