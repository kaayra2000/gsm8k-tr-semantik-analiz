# Hesaplamalı Anlambilim - Semantik Benzerlik Analizi

Bu proje, Türkçe soru-cevap çiftleri arasındaki semantik benzerliği farklı dil modellerini kullanarak analiz eder. Üst dizindeki `gsm8k_tr_1000_soru_cevap.csv` veri kümesini kullanarak sorulara en benzer cevapları ve cevaplara en benzer soruları bulup modellerin performansını değerlendirir.

## Proje Amacı

- Sorulara en yakın 5 cevabı bularak modelin top1 ve top5 başarısını ölçmek
- Cevaplara en yakın 5 soruyu bularak modelin top1 ve top5 başarısını ölçmek
- Metin gömmelerini t-SNE ile görselleştirmek
- Farklı dil modellerinin semantik benzerlik performansını karşılaştırmak

## Kullanılan Modeller

Varsayılan olarak aşağıdaki modeller kullanılmıştır:

1. `ibm-granite/granite-embedding-107m-multilingual` (107M parametre)
2. `intfloat/multilingual-e5-small` (118M parametre)
3. `HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1` (494M parametre)
4. `Alibaba-NLP/gte-multilingual-base` (305M parametre)
5. `intfloat/multilingual-e5-large-instruct` (560M parametre)
6. `ytu-ce-cosmos/turkish-colbert`

Eğer farklı modeller kullanmak isterseniz, `main.py` dosyasındaki `model_names` değişkenini güncelleyebilirsiniz.

## Dosya Yapısı

### Ana Dosyalar

- `main.py`: Tüm işlemi koordine eden ana script
- `gomme_islemleri.py`: Metin gömmelerine ve t-SNE dönüşümüne ilişkin fonksiyonlar
- `benzerlik_islemleri.py`: Metin benzerliği hesaplama fonksiyonları
- `basari_hesapla.py`: Top1 ve Top5 başarı ölçümlerine ilişkin fonksiyonlar
- `gorsellestir.py`: Sonuçları görselleştirme fonksiyonları
- `dosya_islemleri.py`: Dosya yükleme/kaydetme işlemleri için yardımcı fonksiyonlar
- `gereksinimler.txt`: Gerekli Python paketlerinin listesi

## Dosya İçerikleri ve Fonksiyonları

### `main.py`
Ana çalıştırma dosyasıdır. Tüm modeller için benzerlik hesaplama, değerlendirme ve görselleştirme işlemlerini koordine eder.

Önemli fonksiyonlar:
- `generate_similarity_json()`: Kaynak metinle hedef metinler arasındaki benzerliği hesaplar
- `main()`: Ana yürütme fonksiyonu, tüm işlem akışını kontrol eder

### `gomme_islemleri.py` 
Metinlerin gömme vektörlerini hesaplar ve t-SNE uygulamasını gerçekleştirir.

Önemli fonksiyonlar:
- `get_token_embeddings()`: Verilen metni tokenize eder ve her token için gömme vektörlerini çıkarır
- `apply_tsne()`: Gömme vektörlerini 2 boyuta indirgeyerek görselleştirmeye hazırlar

### `benzerlik_islemleri.py`
Metinler arasındaki benzerlik hesaplamalarını yapar.

Önemli fonksiyonlar:
- `get_cosine_similarity()`: İki gömme vektörü arasındaki kosinüs benzerliğini hesaplar
- `find_top5_similar()`: Verilen metne en benzer 5 metni bulur

### `basari_hesapla.py`
Modellerin top1 ve top5 başarı metriklerini hesaplar.

Önemli fonksiyonlar:
- `evaluate_similarity_results_top1_top5()`: Benzerlik sonuçlarının doğruluğunu değerlendirir

### `gorsellestir.py`
Sonuçları görsel olarak sunmaya yarar.

Önemli fonksiyonlar:
- `plot_two_tsne_results()`: İki metin kümesinin t-SNE dönüşümlerini görselleştirir
- `visualize_top1_top5_scores()`: Tüm modellerin top1 ve top5 skorlarını çubuk grafik olarak gösterir

### `dosya_islemleri.py`
Dosya yükleme, kaydetme ve dönüştürme işlemlerini gerçekleştirir.

Önemli fonksiyonlar:
- `load_dataset()`: CSV veri setini yükler
- `load_model()`: HuggingFace modellerini yükler
- `save_smilarity_json()`: Benzerlik sonuçlarını JSON formatında kaydeder
- `save_top1_top5_results_json()`: Top1/Top5 değerlendirme sonuçlarını kaydeder
- `get_all_top1_top5_results()`: Tüm modellerin sonuçlarını yükleme işlemini gerçekleştirir

## Kurulum ve Çalıştırma

### Gereksinimler

Aşağıdaki paketlerin yüklü olması gerekmektedir:
```bash
pip install -r gereksinimler.txt
```
Temel gereksinimler:
* transformers
* torch
* numpy
* pandas
* scikit-learn
* matplotlib

### Veri Kümesi

Kök dizinde `gsm8k_tr_1000_soru_cevap.csv` dosyasının bulunduğundan emin olun. Bu dosya, 1000 adet Türkçe soru-cevap çiftini içermelidir.

### Çalıştırma

Proje klasöründe aşağıdaki komutu çalıştırın:

```bash
python main.py
```
### Çıktılar

#### Benzerlik Sonuçları

Her model için soru-cevap ve cevap-soru yönünde benzerlik hesaplamaları JSON formatında `similarity_results/` dizinine kaydedilir. Bu dosyalar aşağıdaki yapıda bilgiler içerir:
```json
{
  "0": {
    "source_text": "...",
    "real_target": "...",
    "top5_texts": ["...", "...", ...],
    "source_text_tsne_embedding": [...],
    "top5_matches": [
      {
        "rank": 1,
        "index": 5,
        "text": "...",
        "score": 0.92,
        "tsne_embedding": [...]
      },
      ...
    ]
  },
  ...
}
```

#### Top1 ve Top5 Sonuçları

Her model için top1 ve top5 başarı metrikleri `top1_top5_results/` dizinine kaydedilir:

```json
{
  "model_name": "intfloat/multilingual-e5-small",
  "result_type": "question_to_answer",
  "top1_score": {
    "correct": 3,
    "total": 5,
    "accuracy": 0.6,
    "correct_items": [0, 1, 4]
  },
  "top5_score": {
    "correct": 5,
    "total": 5,
    "accuracy": 1.0,
    "correct_items": [[0, 1], [1, 1], [2, 3], [3, 2], [4, 1]]
  }
}
```

#### Görselleştirmeler

**t-SNE Görselleştirmeleri**

Her model için soru ve cevap gömmelerinin 2 boyutlu gösterimi `similarity_results/gorseller/` dizininde saklanır. Bu görseller, soru ve cevap vektörlerinin semantik uzaydaki dağılımını gösterir.

**Top1 ve Top5 Çubuk Grafikleri**
Tüm modellerin top1 ve top5 başarılarını gösteren çubuk grafikler `top1_top5_results/gorseller/` dizininde saklanır:

* top1_top5_results_question_to_answer.png: Soru->Cevap yönündeki başarıları gösterir
* top1_top5_results_answer_to_question.png: Cevap->Soru yönündeki başarıları gösterir