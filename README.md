# 🛡️ Kredi Kartı Dolandırıcılık Tespiti

Bu proje, makine öğrenmesindeki en zorlu problemlerden biri olan **dengesiz veri setleri (imbalanced datasets)** ile başa çıkmak için tasarlanmıştır. 
Yüz binlerce yasal finansal işlem arasına gizlenmiş çok az sayıdaki sahtekarlık (fraud) vakasını tespit etmek amacıyla, **Nesne Yönelimli Programlama (OOP)** standartlarına uygun olarak
modüler bir yapıda geliştirilmiştir.

Dengesiz verilerde standart sınıflandırma algoritmaları yerine, farklılıkları tespit etmede çok daha güçlü bir yaklaşım
olan **Isolation Forest (İzolasyon Ormanı)** tabanlı Anormallik Tespiti (Anomaly Detection) modeli kullanılmıştır.

## 🚀 Projenin Öne Çıkan Özellikleri

*   **Otomatik Veri Tedariği:** Manuel dosya indirmeye gerek yoktur. Sistem, `DataDownloader` sınıfı sayesinde OpenML üzerinden gerçek kredi kartı veri setini otomatik çeker ve yönetir.
*   **Modüler OOP Mimarisi:** Veri indirme, ön işleme, modelleme ve değerlendirme aşamalarının tamamı ayrı sınıflara (`class`) bölünerek temiz, sürdürülebilir ve endüstri standartlarında bir mimari kurulmuştur.
*   **Denetimsiz Öğrenme (Unsupervised):** Model `fit()` aşamasında cevap anahtarı (y_train) olmadan eğitilmiş, veri noktalarının dağılımına bakarak kendi kendine izole/anormal işlemleri tespit etmeyi öğrenmiştir.
*   **İleri Düzey Metrik Ölçümü:** %99.8'i normal olan bir veri setinde yanıltıcı bir metrik olan 'Accuracy' (Doğruluk) yerine, modelin gerçek dolandırıcıları ne oranda yakaladığını gösteren **Recall (Duyarlılık)** metriğine odaklanılmış; sonuçlar Seaborn ile görselleştirilmiştir.

## 📂 Proje Mimarisi

```text
Kredi-Karti-Dolandiricilik-Tespiti/
├── src/
│   ├── __init__.py
│   ├── data_downloader.py  # OpenML API'den veriyi indirir ve kaydeder.
│   ├── data_processor.py   # StandardScaler ile veriyi ölçekler ve Train/Test olarak böler.
│   ├── model_handler.py    # Isolation Forest algoritmasının yapılandırılması ve eğitimi.
│   └── evaluator.py        # Classification Report ve Confusion Matrix (Isı Haritası) çıktısı üretir.
├── main.py                 # Sınıfları başlatan ve projenin akışını yöneten ana orkestra şefi.
├── requirements.txt        # Projenin çalışması için gerekli kütüphaneler listesi.
└── .gitignore              # Veri setini ve sanal ortam klasörlerini GitHub'dan izole eder.
```

## ⚙️ Kurulum ve Çalıştırma

1. Projeyi bilgisayarınıza klonlayın:

```bash
git clone [https://github.com/Abidin-Isik-Yilmazer/Kredi-Karti-Dolandiricilik-Tespiti.git](https://github.com/Abidin-Isik-Yilmazer/Kredi-Karti-Dolandiricilik-Tespiti.git)
```

2. Proje dizinine gidin:

```bash
cd Kredi-Karti-Dolandiricilik-Tespiti
```

3. Gerekli kütüphaneleri yükleyin:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

4. Projeyi çalıştırın:

```bash
python main.py
```

## 📊 Sonuçlar ve Performans

Kod çalıştırıldığında, modelin test seti üzerindeki performansını detaylandıran bir terminal raporu ve görsel bir **Karmaşıklık Matrisi (Confusion Matrix)** ekrana gelir.

Dengesiz veri şartlarında ve tamamen denetimsiz öğrenme (unsupervised) uygulanan bu senaryoda;
*   Sistemdeki 56.864 normal işlem yüksek doğrulukla sorunsuz onaylanmıştır.
*   Veri setinde gizlenmiş **98 gerçek dolandırıcılık vakasının %35'i (34 adet)** sadece verideki anormallik özellikleri izlenerek başarıyla tespit edilmiştir.
