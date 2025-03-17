from typing import Dict, Tuple, List, TYPE_CHECKING
import json
import os
import pandas as pd
from transformers import AutoModel, AutoTokenizer
if TYPE_CHECKING:
    import numpy as np

file_dir = os.path.dirname(os.path.abspath(__file__))
similarity_results_dir = os.path.join(file_dir, "similarity_results")
top1_top5_results_dir = os.path.join(file_dir, "top1_top5_results")
embeddings_dir = os.path.join(file_dir, "embeddings")
tsne_save_dir = os.path.join(file_dir, "tsne_results")
probabilities_dir = os.path.join(file_dir, "probabilities")
def tr_to_lower(text: str) -> str:
    """
    Verilen metni Türkçe karakterleri doğru şekilde koruyarak küçük harfe çeviren fonksiyon.
    """
    # Türkçe büyük karakterlerin küçük karşılıkları
    tr_map = {
        'İ': 'i',
        'I': 'ı',
        'Ğ': 'ğ',
        'Ü': 'ü',
        'Ö': 'ö',
        'Ş': 'ş',
        'Ç': 'ç'
    }
    
    # Önce Türkçe karakterleri dönüştür
    for upper, lower in tr_map.items():
        text = text.replace(upper, lower)
    
    # Sonra normal küçültme işlemini yap
    return text.lower()
def get_result_infix(is_question_to_answer: bool) -> str:
    """
    Sonuç dosyalarının adında kullanılacak ek getiren fonksiyon.

    Args:
        is_question_to_answer: Soru-cevap benzerliği mi yoksa cevap-soru benzerliği mi olduğu
    
    Returns:
        str: Sonuç dosyalarının adında kullanılacak ek
    """
    return "question_to_answer" if is_question_to_answer else "answer_to_question"
def get_top1_top5_result_file_name(prefix: str, is_question_to_answer: bool) -> str:
    """
    Top1 ve top5 sonuçlarının kaydedileceği JSON dosyasının adını döndüren fonksiyon.

    Args:
        prefix: Model adının yer aldığı dosya adı öneki
        is_question_to_answer: Soru-cevap benzerliği mi yoksa cevap-soru benzerliği mi olduğu

    Returns:
        str: JSON dosyasının adı
    """
    return f"{prefix}_{get_result_infix(is_question_to_answer)}_top1_top5_results.json"
def save_top1_top5_results_json(data: Dict, prefix: str, is_question_to_answer: bool):
    """
    Verilen veriyi JSON formatında kaydeden fonksiyon.
    
    Args:
        data: Kaydedilecek veri
        prefix: Model adının yer aldığı dosya adı öneki
        is_question_to_answer: Soru-ceva benzerliği mi yoksa cevap-soru benzerliği mi olduğu
    """
    if not os.path.exists(top1_top5_results_dir):
        os.makedirs(top1_top5_results_dir)
    file_name = get_top1_top5_result_file_name(prefix, is_question_to_answer)
    file_path = os.path.join(top1_top5_results_dir, file_name)
    with open(file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
def get_all_top1_top5_results() -> List[Dict]:
    """
    Tüm top1 ve top5 sonuçlarını okuyan fonksiyon.
    
    Returns:
        Dict: Okunan JSON verisi
    """
    all_results = []
    for file_name in os.listdir(top1_top5_results_dir):
        if file_name.endswith(".json"):
            with open(os.path.join(top1_top5_results_dir, file_name), 'r') as f:
                data = json.load(f)
                all_results.append(data)
    return all_results
def get_probabilities_dir(prefix: str) -> str:
    """
    Modelin olasılıklarını kaydedileceği dizini döndüren fonksiyon.
    
    Args:
        prefix: Model adının yer aldığı dosya adı öneki
    
    Returns:
        str: Olasılıkların kaydedileceği dizin
    """
    return os.path.join(probabilities_dir, f"{prefix}_probabilities.json")

def get_calculated_probabilities_size(prefix: str) -> int:
    """
    prefix'e göre kaydedilen olasılık vektörlerinin sayısını döndüren fonksiyon.
    
    Args:
        prefix: Kaydedilecek dosya adının öneki
    
    Returns:
        int: Dosyada şimdiye kadar kaç olasılık vektörü kaydedildiği
    """
    file_path = get_probabilities_dir(prefix)
    if not os.path.exists(file_path):
        return 0
    return sum(1 for line in open(file_path))

def append_probability(prefix: str, index1: int, index2: int, probability: float,
                       source_dest_type: str = "question_to_answer"):
    """
    Bir olasılık vektörünü ve ilgili veriyi JSON formatında dosyaya ekleyen fonksiyon.
    Her bir olasılık vektörü yeni bir satırda kaydedilir.
    
    Args:
        prefix: Kaydedilecek dosya adının öneki
        item: Kaydedilecek pandas Series nesnesi (soru ve cevap içeren)
        probabilities: Olasılık vektörü
    """
    # Eğer dosya yoksa oluştur
    if not os.path.exists(probabilities_dir):
        os.makedirs(probabilities_dir)
    
    # Dosya yolu
    file_path = get_probabilities_dir(prefix)

    # JSON objesi oluştur
    json_object = {
        "index1": index1,
        "index2": index2,
        "probability": probability,
        "source_dest_type": source_dest_type
    }

    # Dosyayı aç ve yeni girişi ekle
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    with open(file_path, 'a' if file_exists else 'w', encoding='utf-8') as f:
        # JSON objesini ve virgülü tek bir satıra yaz
        f.write(json.dumps(json_object, ensure_ascii=False))
        f.write(',\n')
def read_probability_from_file(prefix: str) -> list:
    """
    JSON formatında kaydedilmiş olasılık vektörlerini ve ilgili verileri okuyan fonksiyon.
    
    Args:
        prefix: Kaydedilen dosya adının öneki

    Returns:
        list: Soru, cevap ve olasılık vektörleri içeren nesnelerin listesi
    """
    # Dosya yolu
    file_path = get_probabilities_dir(prefix)
    
    # Dosya var mı diye kontrol et
    if not os.path.exists(file_path):
        print(f"Dosya bulunamadı: {file_path}")
        return []
    
    # Tüm olasılıkları oku
    probabilities = []
    with open(file_path, 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            line_num += 1
            # Boş satırları atla
            line = line.strip()
            if not line:
                continue
                
            # Satırın sonundaki virgülü kaldır
            if line.endswith(','):
                line = line[:-1]
                
            # JSON nesnesini ayrıştır
            try:
                probability_obj = json.loads(line)
                probabilities.append(probability_obj)
            except json.JSONDecodeError as e:
                print(f"Satır {line_num} işlenemedi: {e}")
                continue
    
    print(f"{os.path.basename(file_path)} dosyasından {len(probabilities)} olasılık vektörü yüklendi.")
    return probabilities
def read_top1_top5_results_json(prefix: str, is_question_to_answer: bool) -> Dict:
    """
    JSON dosyasından top1 ve top5 sonuçlarını okuyan fonksiyon.
    
    Args:
        prefix: Model adının yer aldığı dosya adı öneki
        is_question_to_answer: Soru-cevap benzerliği mi yoksa cevap-soru benzerliği mi olduğu
    
    Returns:
        Dict: Okunan JSON verisi
    """
    file_name = get_top1_top5_result_file_name(prefix, is_question_to_answer)
    file_path = os.path.join(top1_top5_results_dir, file_name)
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_similarity_json_file_name(prefix: str, is_question_to_answer: bool) -> str:
    """
    Benzerlik sonuçlarının kaydedileceği JSON dosyasının adını döndüren fonksiyon.

    Args:
        prefix: Model adının yer aldığı dosya adı öneki
        is_question_to_answer: Soru-cevap benzerliği mi yoksa cevap-soru benzerliği mi olduğu

    Returns:
        str: JSON dosyasının adı
    """
    return f"{prefix}_{get_result_infix(is_question_to_answer)}_similarity.json"
def read_similarity_json(prefix: str, is_question_to_answer: bool) -> Dict:
    """
    JSON dosyasından benzerlik verisini okuyan fonksiyon.

    Args:
        prefix: Model adının yer aldığı dosya adı öneki
        is_question_to_answer: Soru-cevap benzerliği mi yoksa cevap-soru benzerliği mi olduğu
    
    Returns:
        Dict: Okunan JSON verisi
    """
    file_name = get_similarity_json_file_name(prefix, is_question_to_answer)

    file_path = os.path.join(similarity_results_dir, file_name)
    if not os.path.exists(file_path):
        return {}
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

def save_smilarity_json(data: Dict, prefix: str, is_question_to_answer: bool):
    """
    Verilen veriyi JSON formatında kaydeden fonksiyon.

    Args:
        data: Kaydedilecek veri
        prefix: Model adının yer aldığı dosya adı öneki
        is_question_to_answer: Soru-cevap benzerliği mi yoksa cevap-soru benzerliği mi olduğu
    """
    if not os.path.exists(similarity_results_dir):
        os.makedirs(similarity_results_dir)
    file_name = get_similarity_json_file_name(prefix, is_question_to_answer)
    file_path = os.path.join(similarity_results_dir, file_name)
    with open(file_path, 'w') as f:
         json.dump(data, f, ensure_ascii=False, indent=4)


def load_model(model_name: str, device_type: str = "cuda") -> Tuple[AutoModel, AutoTokenizer]:
    """
    Hugging Face model ve tokenizer yükleyen fonksiyon.

    Args:
        model_name: Hugging Face model adı
        device_type: Kullanılacak cihaz (cuda ya da cpu)

    Returns:
        model: Yüklenen model
        tokenizer: Yüklenen tokenizer
    """
    print(f"Model yükleniyor: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device_type)
    print(f"Model yüklendi. {model_name}")
    return model, tokenizer


def load_dataset() -> pd.DataFrame:
    """
    Veri kümesini yükleyen fonksiyon.

    Returns:
        df: Yüklenen veri kümesi
    """ 
    # kod dosya yolu
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # veri kümesi dosya yolu
    dataset_path = os.path.join(current_dir, '..', "gsm8k_tr_1000_soru_cevap.csv")
    
    # Veri kümesini yükle
    print(f"Veri kümesi yükleniyor: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # tüm karakterleri küçült
    df['question'] = df['question'].apply(tr_to_lower)
    df['answer'] = df['answer'].apply(tr_to_lower)
    df["index"] = range(0, len(df))  # 1'den başlayarak her satıra sırayla numara ver
    return df

def get_embeddings_path(save_prefix: str) -> str:
    """
    Gömme vektörlerinin kaydedileceği dosya yolunu döndüren fonksiyon.

    Args:
        save_prefix: Kaydedilecek dosya adının öneki

    Returns:
        str: Gömme vektörlerinin kaydedileceği dosya yolu
    """
    return os.path.join(embeddings_dir, f"{save_prefix}_embeddings.json")

def get_calculated_embeddings_size(save_prefix: str) -> int:
    """
    save_prefix'e göre kaydedilen gömme vektörlerinin sayısını döndüren fonksiyon.

    Args:
        save_prefix: Kaydedilecek dosya adının öneki

    Returns:
        int: dosyada şimdiye kadar kaç gömme vektörü kaydedildiği
    """
    file_path = get_embeddings_path(save_prefix)
    if not os.path.exists(file_path):
        return 0
    return sum(1 for line in open(file_path))

def append_embedding(save_prefix: str, item: pd.Series, question_embedding: 'np.ndarray', answer_embedding: 'np.ndarray'):
    """
    Bir gömme vektörünü ve ilgili veriyi JSON formatında dosyaya ekleyen fonksiyon.
    Her bir gömme yeni bir satırda kaydedilir.
    
    Args:
        save_prefix: Kaydedilecek dosya adının öneki
        item: Kaydedilecek pandas Series nesnesi (soru ve cevap içeren)
        question_embedding: Soru gömme vektörü
        answer_embedding: Cevap gömme vektörü
    """
    
    # Create directory if it doesn't exist
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
        
    # Get the file path with json extension
    json_path = get_embeddings_path(save_prefix)
    
    # Create a JSON object with question, answer and embedding
    json_object = {
        "index": int(item["index"]),
        "question": item["question"],
        "answer": item["answer"],
        "question_embedding": question_embedding.tolist(),
        "answer_embedding": answer_embedding.tolist()
    }
    
    # Open file and append the new entry
    file_exists = os.path.exists(json_path) and os.path.getsize(json_path) > 0
    with open(json_path, 'a' if file_exists else 'w', encoding='utf-8') as f:
        # Write the JSON object and a comma on a single line
        f.write(json.dumps(json_object, ensure_ascii=False))
        f.write(',\n')

def read_embedding_from_file(save_prefix: str) -> list:
    """
    JSON formatında kaydedilmiş gömme vektörlerini ve ilgili verileri okuyan fonksiyon.
    
    Args:
        save_prefix: Kaydedilen dosya adının öneki
        
    Returns:
        list: Soru, cevap ve gömme vektörleri içeren nesnelerin listesi
    """
    
    # Get the file path
    json_path = get_embeddings_path(save_prefix)
    
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"Dosya bulunamadı: {json_path}")
        return []
    
    # Read all embeddings
    embeddings = []
    with open(json_path, 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            line_num += 1
            # Boş satırları atla
            line = line.strip()
            if not line:
                continue
                
            # Sondaki virgülü kaldır
            if line.endswith(','):
                line = line[:-1]
                
            # JSON nesnesini ayrıştır
            try:
                embedding_obj = json.loads(line)
                
                # Gömme listesini numpy dizisine dönüştür
                if isinstance(embedding_obj.get('embedding'), list):
                    embedding_obj['embedding'] = np.array(embedding_obj['embedding'])
                    
                embeddings.append(embedding_obj)
            except json.JSONDecodeError as e:
                print(f"Satır {line_num} işlenemedi: {e}")
                continue
    
    print(f"{os.path.basename(json_path)} dosyasından {len(embeddings)} gömme vektörü yüklendi.")
    return embeddings

def get_tsne_file_path(prefix: str) -> str:
    """
    t-SNE sonuçlarının kaydedileceği dosya adını döndüren fonksiyon.
    
    Args:
        prefix: Kaydedilecek dosya adının öneki
    
    Returns:

        str: t-SNE sonuçlarının kaydedileceği dosya adı
    """
    return os.path.join(tsne_save_dir, f"{prefix}_tsne_results.json")
def get_tsne_photo_path(prefix: str) -> str:
    """
    t-SNE sonuçlarının kaydedileceği dosya adını döndüren fonksiyon.
    
    Args:
        prefix: Kaydedilecek dosya adının öneki

    Returns:
    
        str: t-SNE sonuçlarının kaydedileceği dosya adı
    """
    return os.path.join(tsne_save_dir, f"{prefix}_tsne_results.png")

def get_example_tsne_photo_path(prefix: str) -> str:
    """
    Örnek t-SNE sonuçlarının kaydedileceği dosya adını döndüren fonksiyon.
    
    Args:
        prefix: Kaydedilecek dosya adının öneki

    Returns:

        str: t-SNE sonuçlarının kaydedileceği dosya adı
    """
    return os.path.join(tsne_save_dir, "ornekler", f"{prefix}_tsne_results.png")

def save_tsne_json(data: Dict, save_prefix: str):
    """
    t-SNE sonuçlarını JSON formatında kaydeden fonksiyon.
    
    Args:
        data: Kaydedilecek t-SNE sonuçları
        save_prefix: Kaydedilecek dosya adının öneki
    """
    # Create directory if it doesn't exist
    if not os.path.exists(tsne_save_dir):
        os.makedirs(tsne_save_dir)
    
    # Get the file path with json extension
    json_path = get_tsne_file_path(save_prefix)
    
    # Write the data to the file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_tsne_json(save_prefix: str) -> Dict:
    """
    JSON formatında kaydedilmiş t-SNE sonuçlarını okuyan fonksiyon.
    
    Args:
        save_prefix: Kaydedilen dosya adının öneki

    Returns:

        Dict: Okunan t-SNE sonuçları
    """
    # Get the file path
    json_path = get_tsne_file_path(save_prefix)
    
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"Dosya bulunamadı: {json_path}")
        return {}
    
    # Read the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"{os.path.basename(json_path)} dosyasından t-SNE sonuçları yüklendi.")
    return data