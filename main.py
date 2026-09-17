from src.data_downloader import DataDownloader
from src.data_processor import DataProcessor
from src.model_handler import ModelHandler
from src.evaluator import Evaluator  # Değerlendirme sınıfımızı ekledik


def main():
    data_path = "data/creditcard.csv"

    # 1. Veriyi İndir
    downloader = DataDownloader(save_path=data_path)
    downloader.download_data()

    # 2. Veriyi Oku ve Ön İşleme Yap
    processor = DataProcessor(file_path=data_path)
    df = processor.load_data()
    processor.preprocess()

    # Özellikleri ayır ve eğitim/test olarak böl
    X, y = processor.get_features_and_labels()
    X_train, X_test, y_train, y_test = processor.split_data(X, y)

    # 3. Modeli Başlat, Eğit ve Tahmin Yap
    model_handler = ModelHandler(contamination=0.002)
    model_handler.train(X_train)
    predictions = model_handler.predict(X_test)

    # 4. Değerlendirme ve Görselleştirme
    evaluator = Evaluator(y_true=y_test, y_pred=predictions)
    evaluator.show_report()
    evaluator.plot_confusion_matrix()


if __name__ == "__main__":
    main()