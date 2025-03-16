from typing import Dict

def evaluate_similarity_results_top1_top5(similarity_dict: Dict, model_name: str, is_question_to_answer: bool) -> Dict:
    """
    Benzerlik sonuçları içeren bir sözlüğü değerlendirerek top1 ve top5 sonuçlarını döndürür.
    
    Args:
        similarity_dict: Benzerlik sonuçları içeren sözlük
        model_name: Model adı
        is_question_to_answer: Soru-cevap benzerliği mi yoksa cevap-soru benzerliği mi olduğu
        
    Returns:
        Dict: Top1 ve top5 doğruluk oranlarını içeren sözlük
    """
    # Sonuç sözlükleri
    top1_dict = {
        "correct": 0,
        "total": 0,
        "accuracy": 0.0,
        "correct_items": []  # Doğru tahmin edilen öğelerin indeksleri
    }
    
    top5_dict = {
        "correct": 0,
        "total": 0,
        "accuracy": 0.0,
        "correct_items": []  # Doğru tahmin edilen öğelerin indeksleri ve rankları
    }
    
    # Her öğe için değerlendirme yap
    for idx, item in similarity_dict.items():
        real_target = item["real_target"]
        top5_matches = item["top5_matches"]
        
        # Top1 değerlendirmesi (rank=1 olan eşleşme doğru mu?)
        top1_match = next((match for match in top5_matches if match["rank"] == 1), None)
        if top1_match and top1_match["text"] == real_target:
            top1_dict["correct"] += 1
            top1_dict["correct_items"].append(idx)
        
        top1_dict["total"] += 1
        
        # Top5 değerlendirmesi (herhangi bir eşleşme doğru mu?)
        found_in_top5 = False
        correct_rank = None
        
        for match in top5_matches:
            if match["text"] == real_target:
                found_in_top5 = True
                correct_rank = match["rank"]
                break
        
        if found_in_top5:
            top5_dict["correct"] += 1
            top5_dict["correct_items"].append((idx, correct_rank))
        
        top5_dict["total"] += 1
    
    # Doğruluk oranlarını hesapla
    if top1_dict["total"] > 0:
        top1_dict["accuracy"] = top1_dict["correct"] / top1_dict["total"]
    
    if top5_dict["total"] > 0:
        top5_dict["accuracy"] = top5_dict["correct"] / top5_dict["total"]
    result_type = "question_to_answer" if is_question_to_answer else "answer_to_question"
    
    return {"model_name": model_name, "result_type":result_type, "top1_score": top1_dict, "top5_score": top5_dict}