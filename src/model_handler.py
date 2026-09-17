from sklearn.ensemble import IsolationForest


class ModelHandler:
    def __init__(self, contamination=0.002):
        """
        İzolasyon Ormanı (Isolation Forest) algoritmasını başlatır.
        contamination: Veri setindeki tahmini dolandırıcılık oranıdır.
        Kaggle veri setinde 284 bin işlemde 492 sahtekarlık var (yaklaşık %0.17).
        Biz bunu yuvarlayarak 0.002 (%0.2) veriyoruz.
        """
        # n_estimators: Ormandaki ağaç sayısı
        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=42
        )
        print(f"Isolation Forest modeli başlatıldı. Contamination (Kirlilik) Oranı: {contamination}")

    def train(self, X_train):
        """
        Modeli eğitir.
        DİKKAT: Anomaly Detection 'denetimsiz' (unsupervised) bir yaklaşımdır.
        Bu yüzden .fit() metoduna y_train (cevap anahtarı) VERMİYORUZ.
        Model, sadece X_train içindeki verilerin dağılımına bakarak
        kendi kendine "anormal" (farklı/izole) olanları bulmayı öğrenir.
        """
        print("Model eğitiliyor... (Bu işlem veri setinin büyüklüğüne göre 1-2 dakika sürebilir)")
        self.model.fit(X_train)
        print("Model eğitimi tamamlandı!")

    def predict(self, X):
        """
        Eğitilmiş model ile yeni veriler üzerinde tahmin yapar.
        """
        print("Tahminler üretiliyor...")
        predictions = self.model.predict(X)

        # ÖNEMLİ DÖNÜŞÜM:
        # Isolation Forest algoritması normal işlemlere "1", anormallere (dolandırıcılık) "-1" döner.
        # Ancak bizim orijinal veri setimizde dolandırıcılar "1", normaller "0" olarak işaretli.
        # Metrikleri doğru ölçebilmek için -1'leri 1'e, 1'leri 0'a çeviriyoruz.
        formatted_predictions = [1 if x == -1 else 0 for x in predictions]

        return formatted_predictions