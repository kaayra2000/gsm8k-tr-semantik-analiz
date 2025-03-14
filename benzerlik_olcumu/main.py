import pandas as pd
from transformers import AutoModel, AutoTokenizer
from typing import Dict

from dosya_islemleri import save_smilarity_json, load_model, load_dataset
from benzerlik_islemleri import find_top5_similar


def generate_similarity_json(model: AutoModel, tokenizer: AutoTokenizer, source_column: str,
                            target_column: str, dataset: pd.DataFrame) -> Dict[int, Dict]:
    """
    Veri kümesindeki her kayıt için en benzer 5 kaydı bulan ve JSON formatında döndüren fonksiyon.

    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        source_column: Kaynak kolon (arama yapılacak kolon)
        target_column: Hedef kolon (hangi kolonda arama yapılacağı)
        dataset: Veri kümesi

    Returns:
        result_dict: JSON formatında sonuçlar
    """
    result_dict = {}
    
    for idx, row in dataset.iterrows():
        source_text = row[source_column]
        top5_results, source_text_tsne_embedding = find_top5_similar(model, tokenizer, source_text, target_column, dataset)
        
        # sonuçları JSON formatına dönüştür
        result_dict[idx] = {
            "source_text": source_text,
            "real_target": row[target_column],
            "top5_texts": [match_text for _, _, match_text, _ in top5_results],
            "source_text_tsne_embedding": source_text_tsne_embedding.tolist(),
            "top5_matches": [
                {
                    "index": match_idx,
                    "score": float(score),  # skoru float yap
                    "text": match_text,
                    "tsne_embedding": tsne_embedding.tolist()
                } for match_idx, score, match_text, tsne_embedding in top5_results
            ]
        }
        
        # her 10 elemanda bir yazdır
        if idx % 10 == 0:
            print(f"İşleniyor: {idx+1}/{len(dataset)}")
    print("Tüm elemanlar işlendi.")
    
    return result_dict


def main():
    model_names = [
        "intfloat/multilingual-e5-small",                           # 118M
        "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",     # 494M
        "Alibaba-NLP/gte-multilingual-base",                        # 305M
        "intfloat/multilingual-e5-large-instruct",                  # 560M
        "ytu-ce-cosmos/turkish-colbert"
        ]
    for model_name in model_names:
        save_prefix = model_name.replace("/", "_").replace("-", "_")
        df = load_dataset()
        model, tokenizer = load_model(model_name)
        
        print("Generating question to answer similarity...")
        q_to_a_similarity = generate_similarity_json(model, tokenizer, "question", "answer", df)
        save_smilarity_json(q_to_a_similarity, save_prefix, is_question_to_answer=True)
        
        print("Generating answer to question similarity...")
        a_to_q_similarity = generate_similarity_json(model, tokenizer, "answer", "question", df)
        save_smilarity_json(a_to_q_similarity, save_prefix, is_question_to_answer=False)
    

if __name__ == "__main__":
    main()