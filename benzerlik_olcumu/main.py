import pandas as pd
from typing import Any, Dict, List, Tuple
import json
import os
from ragatouille import RAGPretrainedModel

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

def get_index_from_dataset(content: str, target_column: str, dataset: pd.DataFrame) -> int:
    """
    Verilen metnin veri kümesindeki indeksini döndüren fonksiyon.

    Args:
        content: Aranan metin
        target_column: Hedef kolon (hangi kolonda arama yapılacağı)
        dataset: Veri kümesi
    
    Returns:
        Metnin veri kümesindeki indeksi
    """
    idx = dataset.index[dataset[target_column].apply(tr_to_lower) == content].tolist()[0]
    return idx if idx else -1

def find_top5_similar(model: RAGPretrainedModel, text: str, target_column: str, dataset: pd.DataFrame) -> List[Tuple[int, float]]:
    """
    Verilen metne en benzeyen 5 kaydı döndüren fonksiyon.
    
    Args:
        model: RAGPretrainedModel 
        text: Arama yapılacak metin
        target_column: Hedef kolon (hangi kolonda arama yapılacağı)
        dataset: Veri kümesi
    
    Returns:
        En benzer 5 kaydın (indeks, skor) listesi
    """
    # Metni küçük harfe çevir
    query = tr_to_lower(text)
    
    # Arama yap
    results = model.search(query, k=5)
    
    # Sonuçları işle
    top5_results = []
    for result in results:
        content = result['content']
        score = result.get('score', 0.0)
        top5_results.append((get_index_from_dataset(content, target_column, dataset), score))
    
    return top5_results

def index_model(model: RAGPretrainedModel, target_column: str, dataset: pd.DataFrame) -> None:
    """
        Modeli verilen kolondaki metinlerle indeksleyen fonksiyon.

        Args:
            model: RAGPretrainedModel
            target_column: Hedef kolon (indeksleme yapılacak kolon)
            dataset: Veri kümesi
    """        
    # Metinleri küçük harfe çevir
    print("Metinler küçük harfe dönüştürülüyor...")
    target_corpus = [tr_to_lower(text) for text in dataset[target_column].tolist()]
    
    # Benzersiz indeks adı oluştur
    index_name = f"{target_column}_index"
    
    # Hedef metinleri indeksle
    print(f"{target_column} metinleri indeksleniyor... ({len(target_corpus)} adet)")
    model.index(target_corpus, index_name=index_name)
    print("İndeksleme tamamlandı.")

def format_top5_results(top5_results: List[Tuple[int, float]], dataset: pd.DataFrame, target_column: str) -> List[Dict[str, Any]]:
    """
    En benzer 5 sonucu formatlayan fonksiyon.
    
    Args:
        top5_results: En benzer 5 sonucun listesi
        dataset: Veri kümesi
        target_column: Hedef kolon
    """
    top5_formatted = []
    for rank, (similar_idx, score) in enumerate(top5_results, 1):
        top5_formatted.append({
            "rank": rank,
            "content": dataset[target_column].iloc[similar_idx],
            "score": float(score)
        })
    return top5_formatted

def return_formatted_top5_to_results(source_column: str, source_content: str, target_column: str, top5_formatted: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    En benzer 5 sonucu formatlayıp sonuçlara ekleyen fonksiyon.
    
    Args:
        results: Sonuçlar
        top5_results: En benzer 5 sonucun listesi
        dataset: Veri kümesi
        target_column: Hedef kolon
    """
    return {
            "source_column": source_column,
            "source_content": source_content,
            "target_column": target_column,
            "top5_similars": top5_formatted
        }

def generate_similarity_json(model: RAGPretrainedModel, source_column: str, target_column: str, dataset: pd.DataFrame) -> Dict[int, Dict]:
    """
    Bir kolondaki her eleman için diğer kolondaki en benzer 5 kaydı JSON formatında döndüren fonksiyon.
    
    Args:
        model: RAGPretrainedModel
        source_column: Kaynak kolon (içeriği sorgu olarak kullanılacak)
        target_column: Hedef kolon (içinde benzerlikler aranacak)
        dataset: Veri kümesi
        
    Returns:
        Benzerlik sonuçlarını içeren sözlük
    """
    results = {}
    
    # Modeli indexle
    index_model(model, target_column, dataset)
    
    
    # Her kaynak öğesi için en benzer 5 hedef öğesini bul
    for idx, row in dataset.iterrows():
        print(f"İşleniyor: {idx+1}/{len(dataset)}")
        # bu metnin benzerleri bulunacak
        source_content = row[source_column]
        
        # En benzer 5 kaydı bul
        top5_results = find_top5_similar(model, source_content, target_column, dataset)
        
        # Sonuçları formatla
        top5_formatted = format_top5_results(top5_results, dataset, target_column)
        
        # Bu öğe için sonuçları sakla
        results[int(idx)] = return_formatted_top5_to_results(source_column, source_content, target_column, top5_formatted)
    model.clear_index()
    return results

def main():
    model_name = "ytu-ce-cosmos/turkish-colbert"
    
    # Kod dosyasının bulunduğu dizini alalım
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Veri kümesi yolu
    dataset_path = os.path.join(current_dir, '..', "gsm8k_tr_1000_soru_cevap.csv")
    
    # Veri kümesini oku
    print(f"Veri kümesi yükleniyor: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df = df.head(10)  # Sadece ilk 10 örneği kullan
    print(f"Veri kümesi yüklendi: {len(df)} kayıt")

    print(f"\nModel yükleniyor: {model_name}")
    
    # Model adını dosya adında kullanmak için düzenle
    model_filename = model_name.replace('/', '_').replace('-', '_')
    
    try:
        # Modeli yükle
        model = RAGPretrainedModel.from_pretrained(model_name)
        print(f"Model başarıyla yüklendi: {model_name}")
        
        # Soru-Cevap benzerlikleri hesaplanıyor
        print(f"Soru benzerlikleri hesaplanıyor...")
        similarity_results_path = os.path.join(current_dir, f'similarity_results_{model_filename}.json')
        question_results = generate_similarity_json(model, "question", "answer", df)
        
        # Sonuçları kaydet
        print(f"Sonuçlar kaydediliyor: {similarity_results_path}")
        with open(similarity_results_path, 'w', encoding='utf-8') as f:
            json.dump(question_results, f, ensure_ascii=False, indent=4)
        
        print(f"İşlem tamamlandı: {similarity_results_path}")
        
        # Cevap-Soru benzerlikleri hesaplanıyor
        answer_results_path = os.path.join(current_dir, f'similarity_results_answers_{model_filename}.json')
        print(f"Cevap benzerlikleri hesaplanıyor...")
        answer_results = generate_similarity_json(model, "answer", "question", df)
        
        # Sonuçları kaydet
        with open(answer_results_path, 'w', encoding='utf-8') as f:
            json.dump(answer_results, f, ensure_ascii=False, indent=4)
            
        print(f"İşlem tamamlandı: {answer_results_path}")
        
    except Exception as e:
        print(f"Hata oluştu - {model_name}: {str(e)}")

if __name__ == "__main__":
    main()