import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict
from dosya_islemleri import get_tsne_photo_path, get_example_tsne_photo_path
def plot_two_tsne_results(tsne_result1, tsne_result2, save_prefix, 
                         label1="Kaynak Metin", label2="Modelin Benzer Bulduğu Metin",
                         color1="blue", color2="red",
                         figsize=(18, 8)):
    """
    İki farklı t-SNE sonucunu yan yana ayrı grafiklerde çizdirir.
    
    Args:
        tsne_result1: Birinci t-SNE sonucu (n_tokens1 x 2 boyutunda)
        tsne_result2: İkinci t-SNE sonucu (n_tokens2 x 2 boyutunda)
        save_prefix: Kaydedilecek dosya adının öneki
        label1: Birinci veri seti için etiket
        label2: İkinci veri seti için etiket
        color1: Birinci veri seti için renk
        color2: İkinci veri seti için renk
        figsize: Figür boyutu (genişlik, yükseklik)
    """
    # Liste tipindeki verileri NumPy dizisine dönüştür
    tsne_result1 = np.array(tsne_result1)
    tsne_result2 = np.array(tsne_result2)
    # Figürü ve alt grafikleri oluştur
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    title=f"{label1} ve {label2} Gömmelerinin t-SNE Görselleştirmesi",
    # İlk t-SNE sonucunu sol grafikte çizgiyle bağlanmış noktalar olarak çiz
    ax1.plot(
        tsne_result1[:, 0],  # x koordinatları
        tsne_result1[:, 1],  # y koordinatları
        c=color1,
        alpha=0.7,
        linewidth=1.5,
        marker='o',
        markersize=6
    )
    
    # İkinci t-SNE sonucunu sağ grafikte çizgiyle bağlanmış noktalar olarak çiz
    ax2.plot(
        tsne_result2[:, 0],  # x koordinatları
        tsne_result2[:, 1],  # y koordinatları
        c=color2,
        alpha=0.7,
        linewidth=1.5,
        marker='o',
        markersize=6
    )
    
    # Sol grafik düzenlemeleri
    ax1.set_title(f"{label1} Token Gömmeleri", fontsize=14)
    ax1.set_xlabel("t-SNE Boyut 1", fontsize=12)
    ax1.set_ylabel("t-SNE Boyut 2", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Sağ grafik düzenlemeleri
    ax2.set_title(f"{label2} Token Gömmeleri", fontsize=14)
    ax2.set_xlabel("t-SNE Boyut 1", fontsize=12)
    ax2.set_ylabel("t-SNE Boyut 2", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Genel başlık ekle
    fig.suptitle(title, fontsize=16)
    
    # Grafiği düzenle ve kaydet
    plt.tight_layout()
    
    file_path = get_example_tsne_photo_path(save_prefix)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    plt.savefig(file_path, dpi=300, bbox_inches='tight')

def visualize_top1_top5_scores(results_list: List) -> None:
    """
    Model sonuçlarını içeren sözlük listesini alıp top1 ve top5 skorlarını 
    görselleştirerek kaydeden fonksiyon.
    
    Args:
        results_list: Model sonuçlarını içeren sözlük listesi
    """
    # Sonuçları result_type'a göre grupla
    question_to_answer = []
    answer_to_question = []
    
    for result in results_list:
        if result["result_type"] == "question_to_answer":
            question_to_answer.append(result)
        elif result["result_type"] == "answer_to_question":
            answer_to_question.append(result)
    
    # Çıktı dizinini oluştur
    top1_top5_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "top1_top5_results", "gorseller")
    os.makedirs(top1_top5_results_dir, exist_ok=True)
    
    # Soru Cevap Görselleştirmesi
    if question_to_answer:
        plot_t1_t5_scores(question_to_answer, 
                   os.path.join(top1_top5_results_dir, "top1_top5_results_question_to_answer.png"),
                   "Cevahtan Soru Bulma Performansı")
    
    # Cevap Soru Görselleştirmesi
    if answer_to_question:
        plot_t1_t5_scores(answer_to_question, 
                   os.path.join(top1_top5_results_dir, "top1_top5_results_answer_to_question.png"),
                   "Sorudan Cevap Bulma Performansı")

def plot_t1_t5_scores(results: List[Dict], output_path: str, title: str, top1_color='royalblue', top5_color='lightcoral'):
    """
    Belirli bir result_type için skorları görselleştiren ve kaydeden yardımcı fonksiyon.
    
    Args:
        results: Skor sonuçlarını içeren liste
        output_path: Dosya kaydetme yolu
        title: Grafik başlığı
        top1_color: Top-1 skorlar için renk
        top5_color: Top-5 skorlar için renk
    """
    # Model isimlerini ve skorları çıkar
    models = []
    top1_scores = []
    top5_scores = []
    
    for result in results:
        # Uzun model isimlerini kısalt
        model_name = result["model_name"].split('/')[-1] if '/' in result["model_name"] else result["model_name"]
        models.append(model_name)
        top1_scores.append(result["top1_score"]["accuracy"] * 100)  # Yüzde olarak göster
        top5_scores.append(result["top5_score"]["accuracy"] * 100)  # Yüzde olarak göster
    
    # Grafik boyutunu ayarla
    plt.figure(figsize=(12, 6))
    
    # Bar chart konumlarını ayarla
    bar_width = 0.35
    x = np.arange(len(models))
    
    # Bar chartları çiz
    bars1 = plt.bar(x - bar_width/2, top1_scores, bar_width, label='Top-1', color=top1_color)
    bars2 = plt.bar(x + bar_width/2, top5_scores, bar_width, label='Top-5', color=top5_color)
    
    # Grafik özelliklerini ayarla
    plt.xlabel('Modeller')
    plt.ylabel('Doğruluk (%)')
    plt.title(title)
    plt.xticks(x, models, rotation=20, ha='right')
    plt.ylim(0, 110)  # Skor yüzde olduğu için 0-100 arası + biraz boşluk
    
    # Bar üstlerinde değerleri göster
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1.05), framealpha=0.9, 
               edgecolor='lightgray', fancybox=True)
    plt.tight_layout()
    
    # Grafiği kaydet
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Görsel kaydedildi: {output_path}")

def save_tsne_png(data: Dict, save_prefix: str, q_color='blue', a_color='yellow'):
    """
    t-SNE görselleştirme grafiği oluşturur ve kaydeder.
    
    Args:
        data: 'question_tsne' ve 'answer_tsne' noktalarını içeren sözlük
        save_prefix: Kaydedilen dosya için önek
        q_color: Soru noktaları için renk (varsayılan: mavi)
        a_color: Cevap noktaları için renk (varsayılan: sarı)
    
    Returns:
        Kaydedilen PNG dosyasının yolu
    """
    # Dosya yolunu al
    png_path = get_tsne_photo_path(save_prefix)
    
    # Yeni bir şekil oluştur
    plt.figure(figsize=(10, 8))
    
    # Veri noktalarını çıkar
    q_points = data.get("question_tsne", [])
    a_points = data.get("answer_tsne", [])
    
    # answer_tsne formatını kontrol et ve gerekirse düzelt
    if isinstance(a_points, str):
        try:
            # JSON formatında string olabilir, değilse boş liste kullan
            import json
            a_points = json.loads(a_points)
        except:
            a_points = []
    elif isinstance(a_points, list) and len(a_points) > 0 and not isinstance(a_points[0], list):
        a_points = [a_points]
    
    # Çizilecek nokta olup olmadığını kontrol et
    if not q_points and not a_points:
        raise ValueError("Verilen veride t-SNE noktası bulunamadı")
    
    # Soru noktalarını mavi yuvarlak olarak çiz
    if q_points:
        q_x = [point[0] for point in q_points if len(point) >= 2]
        q_y = [point[1] for point in q_points if len(point) >= 2]
        plt.scatter(q_x, q_y, color=q_color, marker='o', label='Sorular')
    
    # Cevap noktalarını sarı kare olarak çiz
    if a_points:
        # Her noktanın en az 2 değer içerdiğinden emin ol
        valid_points = [p for p in a_points if isinstance(p, list) and len(p) >= 2]
        if valid_points:
            a_x = [point[0] for point in valid_points]
            a_y = [point[1] for point in valid_points]
            plt.scatter(a_x, a_y, color=a_color, marker='s', label='Cevaplar')
    
    # Etiketler ve açıklamaları ekle
    plt.title('t-SNE Görselleştirmesi')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Görüntüyü kaydet
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Görselleştirme kaydedildi: {png_path}")
    return png_path