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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096, padding=True)
    
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

def apply_tsne(token_embeddings: np.ndarray, perplexity: int = 5, max_iter: int = 1000, random_state: int = 42,
               counter=[0]) -> np.ndarray:
    """
    Token gömmelerini t-SNE ile 2 boyuta indirger.
    
    Args:
        token_embeddings: Token gömmelerini içeren matris (token_sayısı x gömme_boyutu)
        perplexity: t-SNE için perplexity parametresi
        n_iter: t-SNE için iterasyon sayısı
        random_state: Tekrarlanabilirlik için rastgele seed değeri
    
    Returns:
        np.ndarray: 2 boyutlu t-SNE dönüşümü (token_sayısı x 2)
    """
    # Token sayısını ve gömme boyutunu al
    n_tokens, embedding_dim = token_embeddings.shape
    
    # Eğer token sayısı çok azsa t-SNE için uygun perplexity değerini ayarla
    if n_tokens < 10:
        perplexity = min(perplexity, n_tokens - 1)
        perplexity = max(2, perplexity)  # En az 2 olsun
        if counter[0] == 0:
            print(f"Token sayısı çok az ({n_tokens}), perplexity değeri {perplexity} olarak ayarlandı.")
            counter[0] += 1
    
    # Eğer sadece bir token varsa t-SNE uygulanamaz
    if n_tokens <= 1:
        print("Tek token için t-SNE uygulanamıyor, rastgele bir nokta döndürülüyor.")
        return np.random.rand(1, 2)
    
    # t-SNE modelini oluştur
    tsne = TSNE(
        n_components=2,         # 2 boyuta indirgeme
        perplexity=perplexity,  # Perplexity değeri
        max_iter=max_iter,          # İterasyon sayısı
        learning_rate='auto',   # Otomatik öğrenme oranı
        init='random',          # Rastgele başlangıç
        random_state=random_state  # Tekrarlanabilirlik için seed
    )
    
    # t-SNE'yi uygula
    embeddings_2d = tsne.fit_transform(token_embeddings)
    return embeddings_2d