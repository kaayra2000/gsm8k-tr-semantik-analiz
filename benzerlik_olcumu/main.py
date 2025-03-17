import pandas as pd
from transformers import AutoModel, AutoTokenizer
from typing import Dict
from dosya_islemleri import save_smilarity_json, load_model, load_dataset, save_top1_top5_results_json, get_all_top1_top5_results
from benzerlik_islemleri import generate_similarity_json
from basari_hesapla import evaluate_similarity_results_top1_top5
from gorsellestir import visualize_top1_top5_scores, plot_two_tsne_results


def main():
    model_names = [
        "intfloat/multilingual-e5-base",                                # 278M  - 81
        "ibm-granite/granite-embedding-107m-multilingual",              # 107M  - 48
        "intfloat/multilingual-e5-small",                               # 118M  - 36
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 118M  - 62
        "shibing624/text2vec-base-multilingual",                        # 118M  - 77 
        "ytu-ce-cosmos/turkish-colbert"
        ]
    df = load_dataset()
    for model_name in model_names:
        save_prefix = model_name.replace("/", "_").replace("-", "_")
        model, tokenizer = load_model(model_name)
        
        for source, target in [("answer", "question"), ("question", "answer")]:
            is_question_to_answer = source == "question" and target == "answer"
            direction_name = f"{source}_to_{target}"
            
            print(f"{direction_name} benzerliği oluşturuluyor...")
            
            # Benzerlik sonuçlarını oluştur
            similarity_results = generate_similarity_json(model, tokenizer, source, target, df)
            
            # Benzerlik JSON'larını kaydet
            save_smilarity_json(similarity_results, save_prefix, is_question_to_answer=is_question_to_answer)
            
            # Top1 ve Top5 sonuçlarını değerlendir
            eval_results = evaluate_similarity_results_top1_top5(similarity_results, model_name, is_question_to_answer=is_question_to_answer)
            
            # Değerlendirme sonuçlarını kaydet
            save_top1_top5_results_json(eval_results, save_prefix, is_question_to_answer=is_question_to_answer)
            
            # t-SNE görselleştirmelerini oluştur ve kaydet (sadece ilk sonuç için deneme amaçlı)
            plot_two_tsne_results(
                similarity_results[0]["source_text_tsne_embedding"], 
                similarity_results[0]["top5_matches"][0]["tsne_embedding"],
                file_name=f"tsne_{save_prefix}_{direction_name}.png"
            )
    visualize_top1_top5_scores(get_all_top1_top5_results())

    

if __name__ == "__main__":
    main()