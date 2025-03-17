import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE
import pandas as pd
from dosya_islemleri import get_calculated_embeddings_size, append_embedding, save_tsne_json

def calculate_and_save_raw_embeddings_from_dataset(model: AutoModel, tokenizer: AutoTokenizer, 
                                                   dataset: pd.DataFrame, save_prefix: str, device_type: str = "cuda") -> np.ndarray:
    """
    Verilen model ve tokenizer kullanarak bir veri kümesindeki metinlerin token gömme vektörlerini hesaplar.

    Args:
        model: Hugging Face transformers kütüphanesinden bir AutoModel nesnesi
        tokenizer: Model ile uyumlu bir AutoTokenizer nesnesi
        dataset: Token gömme vektörleri alınacak veri kümesi
        save_prefix: Kaydedilecek dosya adının öneki
        device_type: Hesaplamanın yapılacağı cihaz tipi ("cuda" veya "cpu")
    
    Returns:
        np.ndarray: veri kümesindeki tüm verilerin token gömme vektörlerini içeren numpy dizisi
    """
    calculated_embeddings_size = get_calculated_embeddings_size(save_prefix)
    print(f"Hali hazırda hesaplanmış gömme vektörleri sayısı: {calculated_embeddings_size}, bu kayıtlar atlanıyor.")
    print("\n")
    for i in range(calculated_embeddings_size, len(dataset)):
        item = dataset.iloc[i]
        question_text = item["question"]
        question_embedding = get_token_embedding(model, tokenizer, question_text, device_type)
        answer_text = item["answer"]
        answer_embedding = get_token_embedding(model, tokenizer, answer_text, device_type)
        append_embedding(save_prefix, item, question_embedding, answer_embedding)
        print(f"{i+1}/{len(dataset)} gömme vektörü hesaplandı ve kaydedildi.", end="\r")

def get_token_embedding(model: AutoModel, tokenizer: AutoTokenizer, text: str, device_type: str = "cuda") -> np.ndarray:
    """
    Verilen model ve tokenizer kullanarak bir metnin tek boyutlu cümle gömme vektörünü hesaplar.
    Token seviyesindeki gömmelerden ortalama alarak tek bir vektör oluşturur.
    
    Args:
        model: Hugging Face transformers kütüphanesinden bir AutoModel nesnesi
        tokenizer: Model ile uyumlu bir AutoTokenizer nesnesi
        text: Gömme vektörü oluşturulacak metin
        device_type: Hesaplamanın yapılacağı cihaz tipi ("cuda" veya "cpu")
    
    Returns:
        np.ndarray: Cümle gömme vektörünü içeren tek boyutlu numpy dizisi
    """
    # Girdi metnini tokenize et
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Girdileri model ile aynı cihaza taşı
    inputs = {k: v.to(device_type) for k, v in inputs.items()}
    
    # Model çıktılarını al - bellek tasarrufu için no_grad kullan
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Gömmeleri (son gizli durumları) al
    # Boyut: [batch_size, sequence_length, hidden_size]
    embeddings = outputs.last_hidden_state
    
    # Token gömmelerinin ortalamasını alarak cümle gömmesi oluştur
    # (1, sequence_length, hidden_size) -> (hidden_size,)
    sentence_embedding = embeddings.squeeze(0).mean(axis=0)
    
    # Numpy dizisine dönüştür ve döndür
    return sentence_embedding.cpu().numpy()

# Tekli eleman için t-SNE hesaplayan fonksiyon
def tsne_tekli(embedding, perplexity=30, n_iter=1000):
    """
    Tek bir gömmeyi (embedding) alıp 2 boyutlu t-SNE dönüştürmesi uygular
    
    Args:
        embedding: Giriş gömme vektörü
        perplexity: t-SNE perplexity parametresi
        n_iter: t-SNE iterasyon sayısı
        
    Returns:
        2 boyutlu t-SNE sonucu (liste olarak)
    """
    # Tek bir vektör için TSNE hesaplanamaz, en az 2 örnek gerekir
    # Bu nedenle aynı vektörden iki tane oluşturup sonra ilkini alacağız
    temp_data = np.vstack([embedding, embedding])
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_result = tsne.fit_transform(temp_data)
    return tsne_result[0].tolist()  # İlk sonucu liste olarak döndür

# Çoklu elemanlar için t-SNE hesaplayan fonksiyon
def tsne_coklu(embeddings, perplexity=30, n_iter=1000):
    """
    Birden fazla gömmeyi alıp 2 boyutlu t-SNE dönüştürmesi uygular
    
    Args:
        embeddings: Giriş gömme vektörleri listesi
        perplexity: t-SNE perplexity parametresi
        n_iter: t-SNE iterasyon sayısı
        
    Returns:
        2 boyutlu t-SNE sonuçları (liste olarak)
    """
    if len(embeddings) < 2:
        # Tek eleman varsa tekli fonksiyonu kullan
        return [tsne_tekli(embeddings[0], perplexity, n_iter)]
    
    # Birden fazla eleman varsa doğrudan t-SNE uygula
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings)-1), 
                n_iter=n_iter, random_state=42)
    tsne_result = tsne.fit_transform(np.array(embeddings))
    return tsne_result.tolist()  # Sonuçları liste olarak döndür

# Sonuçları istenen formatta döndüren fonksiyon
def tsne_sonuc_olustur(veri: list, save_prefix: str) -> dict:
    """
    Verilen soru ve cevap gömmelerini alıp t-SNE sonuçlarını istenen formatta döndürür
    
    Args:
        veri: Soru ve cevap gömmelerini içeren veri örneği
        save_prefix: Gömme vektörlerinin kaydedildiği dosya adının öneki

    Returns:
        İstenen formatta t-SNE sonuçları
    """
    soru_tsne = tsne_coklu([item["question_embedding"] for item in veri])
    cevap_tsne = tsne_coklu([item["answer_embedding"] for item in veri])
    tsne_sonuc = {
        "question_tsne": soru_tsne,
        "answer_tsne": cevap_tsne
    }
    save_tsne_json(tsne_sonuc, save_prefix)
    return tsne_sonuc