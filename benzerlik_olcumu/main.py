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
        
        # Güncellenmiş find_top5_similar fonksiyonu artık doğrudan JSON formatında bir dict döndürür
        similarity_result = {"source_text": source_text, "real_target": row[target_column]}
        find5_res = find_top5_similar(model, tokenizer, source_text, target_column, dataset)
        top5_texts = [res["text"] for res in find5_res["top5_matches"]]
        similarity_result.update({"top5_texts": top5_texts})
        similarity_result.update({"top5_matches": find5_res})
        # Sonuçları düzenle
        result_dict[idx] = similarity_result
        
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
    df = load_dataset()

    for model_name in model_names:
        save_prefix = model_name.replace("/", "_").replace("-", "_")
        model, tokenizer = load_model(model_name)
        
        print("Generating question to answer similarity...")
        q_to_a_similarity = generate_similarity_json(model, tokenizer, "question", "answer", df)
        save_smilarity_json(q_to_a_similarity, save_prefix, is_question_to_answer=True)
        
        print("Generating answer to question similarity...")
        a_to_q_similarity = generate_similarity_json(model, tokenizer, "answer", "question", df)
        save_smilarity_json(a_to_q_similarity, save_prefix, is_question_to_answer=False)
    

if __name__ == "__main__":
    main()