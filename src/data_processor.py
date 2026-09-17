import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        print(f"Veri yükleniyor: {self.file_path}")
        self.data = pd.read_csv(self.file_path)
        print(f"Veri yüklendi. Toplam kayıt: {len(self.data)}")
        return self.data

    def preprocess(self):
        # OpenML verisinde Time sütunu olmadığı için sadece Amount'u işliyoruz.
        print("Veri ön işleme yapılıyor (Amount ölçeklendiriliyor)...")
        scaler = StandardScaler()

        self.data['scaled_amount'] = scaler.fit_transform(self.data['Amount'].values.reshape(-1, 1))

        self.data.drop(['Amount'], axis=1, inplace=True)
        return self.data

    def get_features_and_labels(self):
        y = self.data['Class']
        X = self.data.drop('Class', axis=1)
        return X, y

    def split_data(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print("Eğitim ve test setleri başarıyla ayrıldı.")
        return X_train, X_test, y_train, y_test