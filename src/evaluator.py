from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, y_true, y_pred):
        """
        Sınıf başlatıldığında, gerçek sonuçları (y_true) ve modelin tahminlerini (y_pred) alır.
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def show_report(self):
        """
        Precision (Kesinlik), Recall (Duyarlılık) ve F1-Score metriklerini ekrana yazdırır.
        """
        print("\n--- MODEL DEĞERLENDİRME RAPORU ---")
        print(classification_report(self.y_true, self.y_pred))

    def plot_confusion_matrix(self):
        """
        Sonuçları görselleştirmek için Confusion Matrix (Karmaşıklık Matrisi) çizer.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)

        plt.figure(figsize=(8, 6))
        # cmap='Reds' ile dolandırıcılık temasına uygun kırmızı bir ısı haritası çiziyoruz.
        # annot=True parametresi, kutuların içine sayıların yazılmasını sağlar.
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['Normal (0)', 'Dolandırıcı (1)'],
                    yticklabels=['Normal (0)', 'Dolandırıcı (1)'])

        plt.title('Karmaşıklık Matrisi (Confusion Matrix)')
        plt.ylabel('Gerçek İşlem Türü')
        plt.xlabel('Modelin Tahmini')
        plt.show()  # Grafiği ekranda açar