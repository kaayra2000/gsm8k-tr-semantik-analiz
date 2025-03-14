import matplotlib.pyplot as plt
import numpy as np

def plot_two_tsne_results(tsne_result1, tsne_result2, 
                         label1="Kaynak Metin", label2="Modelin Benzer Bulduğu Metin",
                         color1="blue", color2="red",
                         figsize=(18, 8), save_path=None):
    """
    İki farklı t-SNE sonucunu yan yana ayrı grafiklerde çizdirir.
    
    Args:
        tsne_result1: Birinci t-SNE sonucu (n_tokens1 x 2 boyutunda)
        tsne_result2: İkinci t-SNE sonucu (n_tokens2 x 2 boyutunda)
        label1: Birinci veri seti için etiket
        label2: İkinci veri seti için etiket
        color1: Birinci veri seti için renk
        color2: İkinci veri seti için renk
        figsize: Figür boyutu (genişlik, yükseklik)
        save_path: Grafiğin kaydedileceği dosya yolu
    """
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt