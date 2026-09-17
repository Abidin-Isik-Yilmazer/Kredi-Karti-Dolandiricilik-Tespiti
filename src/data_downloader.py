import os
import pandas as pd
from sklearn.datasets import fetch_openml


class DataDownloader:
    def __init__(self, save_path):
        """
        Verinin nereye kaydedileceğini (save_path) tanımlıyoruz.
        """
        self.save_path = save_path

    def download_data(self):
        """
        Scikit-learn üzerinden açık kaynaklı Credit Card Fraud veri setini indirir.
        Kaggle API anahtarlarıyla uğraşmadan veriyi doğrudan çeker.
        """
        # Eğer dosya zaten varsa tekrar indirmemek için kontrol edelim
        if os.path.exists(self.save_path):
            print(f"Veri seti zaten mevcut: {self.save_path}. İndirme atlanıyor.")
            return

        print(
            "Veri seti indiriliyor... (Dosya boyutu büyük, internet hızınıza göre birkaç dakika sürebilir, lütfen bekleyin...)")

        # OpenML platformundan 1597 ID'li gerçek Kredi Kartı Dolandırıcılık veri setini çekiyoruz
        dataset = fetch_openml(data_id=1597, as_frame=True, parser='auto')
        df = dataset.frame

        # 'Class' sütunu OpenML'de metin (string) olarak gelebiliyor, onu sayıya (integer) çevirelim
        df['Class'] = df['Class'].astype(int)

        # Veriyi data klasörüne CSV olarak kaydediyoruz
        df.to_csv(self.save_path, index=False)
        print(f"Harika! Veri seti başarıyla indirildi ve {self.save_path} olarak kaydedildi.")