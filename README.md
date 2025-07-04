 Face Mask Detection with Transfer Learning

Bu proje, insan yüzü fotoğraflarında maske takılı mı değil mi tespit eden bir görüntü sınıflandırma modelidir.
Transfer Learning yöntemi kullanılarak MobileNetV2 önceden eğitilmiş ağıyla model oluşturulmuştur.

Kullanılan Teknolojiler
Python 3.11

TensorFlow 2.x (Keras)

scikit-learn

Matplotlib

NumPy

Veri Seti
Kaggle Link: Face Mask Dataset

Sınıflar:

😷 With Mask

😮 Without Mask

Görseller 224x224 boyutuna getirilmiş ve 0-1 aralığında normalize edilmiştir.

Veriler %80 eğitim, %20 doğrulama olarak ayrılmıştır.

Model Mimarisi
Katman	Özellikler
MobileNetV2	(Pretrained, Frozen)
Global Average Pooling 2D	
Dense	128 nöron, ReLU
Dropout	%30
Dense (Çıkış)	1 nöron, Sigmoid

Loss Function: Binary Crossentropy
Optimizer: Adam
Metric: Accuracy

Eğitim Sonuçları

Performans Grafikleri
 Loss Grafiği -Accuracy Grafiği
![Figure_1](https://github.com/user-attachments/assets/ce3442e2-4db0-4d76-ac05-af33a4c85877)

Confusion Matrix & Sınıflandırma Raporu
![Figure_2](https://github.com/user-attachments/assets/86b97b41-0688-4007-8771-a134d8801f7b)

![Figure_3](https://github.com/user-attachments/assets/92509e4f-529f-4c6d-af49-584252157575)

Örnek Tahminler
![Figure_4](https://github.com/user-attachments/assets/4a9d741e-c89a-40b3-8242-d4cc353d73bf)


Karşılaşılan Zorluklar ve Çözümler (Sorun/Çözüm)
TensorFlow kurulumu uyumsuzluğu/Sanal ortam kuruldu, sürüm düzeltildi
Windows’ta Türkçe klasör yolları sorunu/Yol ayarları düzenlendi
Veri yolu hatası/flow_from_directory kullanıldı.

Sonuç
Transfer learning kullanılarak maske tespiti problemi yüksek doğrulukla çözüldü.
Model, gerçek zamanlı uygulamalara uygun hız ve başarı oranına sahiptir.

Çalıştırma Adımları

Ortamı kur:
python -m venv maskenv
maskenv\Scripts\activate

Gerekli kütüphaneleri yükle:
pip install tensorflow scikit-learn matplotlib

Modeli eğit ve test et:
python mask.py
