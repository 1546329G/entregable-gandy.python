import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Cargar el dataset Fashion MNIST (ropa, no se usó en clase)
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalización: valores entre 0 y 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Agregar canal (porque CNN espera imágenes 3D)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 2. Crear la red convolucional
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 clases
])

# 3. Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Entrenar
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# 5. Evaluar
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n🎯 Precisión en el set de prueba: {test_acc:.2f}")

# 6. Graficar entrenamiento
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.title("Gráfico de precisión del modelo")
plt.show()

# 7. Hacer una predicción
import numpy as np

predictions = model.predict(x_test)
index = 0
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title(f"Predicción: {np.argmax(predictions[index])}")
plt.show()
