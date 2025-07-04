import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

data_dir = Path(r'C:\Users\oznur\OneDrive\Masaüstü\GYK-YapayZeka\CV\Mask\data')
print(data_dir.exists())

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
# Pretrained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze

# Üst katmanlar
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Derleme
model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

def plot_training_history(history):
    # Loss grafiği
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Eğitim Loss')
    plt.plot(history.history['val_loss'], label='Doğrulama Loss')
    plt.title('Loss Grafiği')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Eğitim Accuracy')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Accuracy')
    plt.title('Accuracy Grafiği')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Kullanımı:
plot_training_history(history)

# Doğrulama verisini al
val_generator.reset()
y_true = val_generator.classes
# Tahminler (0-1 arası)
y_pred_prob = model.predict(val_generator)
# 0 veya 1 olarak sınıflandır
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=val_generator.class_indices.keys())

cmd.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Detaylı rapor
print(classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys()))

class_names = list(val_generator.class_indices.keys())

# 5 örnek al
x_test, y_test = next(val_generator)

plt.figure(figsize=(15, 7))

for i in range(5):
    img = x_test[i]
    true_label = class_names[int(y_test[i])]
    pred_prob = model.predict(img[np.newaxis, ...])[0][0]
    pred_label = class_names[int(pred_prob > 0.5)]

    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f"Gerçek: {true_label}\nTahmin: {pred_label}\nProb: {pred_prob:.2f}")
    plt.axis('off')

plt.show()