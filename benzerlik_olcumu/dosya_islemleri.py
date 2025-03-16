from typing import Dict, Tuple
import json
import os
import pandas as pd
from transformers import AutoModel, AutoTokenizer

similarity_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "similarity_results")
top1_top5_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "top1_top5_results")

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


def load_model(model_name: str) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Hugging Face model ve tokenizer yükleyen fonksiyon.

    Args:
        model_name: Hugging Face model adı

    Returns:
        model: Yüklenen model
        tokenizer: Yüklenen tokenizer
    """
    print(f"Model yükleniyor: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    print("Model yüklendi.")
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
    return df