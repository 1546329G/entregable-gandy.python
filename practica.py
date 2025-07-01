import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Función para extraer características avanzadas
def extraer_caracteristicas_avanzadas(imagen):
    mini = cv2.resize(imagen, (100, 100))
    gray = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY)

    # Color
    color_mean = mini.mean(axis=(0, 1))
    color_std = mini.std(axis=(0, 1))
    median_color = np.median(mini.reshape(-1, 3), axis=0)

    # Textura
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edge_count = np.count_nonzero(cv2.Canny(gray, 100, 200))

    # Forma
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(max(contours, key=cv2.contourArea)) if contours else 0

    return np.concatenate([
        color_mean, color_std, median_color,
        [lap_var, edge_count, area]
    ])

# Función para capturar imágenes de la webcam
def capturar_datos(objeto, cantidad, etiqueta, datos, etiquetas):
    cap = cv2.VideoCapture(0)
    print(f"Captura {cantidad} imágenes del objeto: {objeto}")
    capturas = 0

    while capturas < cantidad:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow(f"Captura {objeto} (Presiona ESPACIO)", frame)
        key = cv2.waitKey(1)

        if key == 32:  # Tecla ESPACIO
            feature = extraer_caracteristicas_avanzadas(frame)
            datos.append(feature)
            etiquetas.append(etiqueta)
            capturas += 1
            print(f"{objeto} capturado: {capturas} / {cantidad}")
        elif key == 27:  # Tecla ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# Captura de datos
X = []
y = []

# Captura 20 imágenes por objeto (puedes aumentar si deseas)
capturar_datos("celular", 10, "celular", X, y)
capturar_datos("rostro", 10, "rostro", X, y)
capturar_datos("llaves",10, "llaves", X, y)
capturar_datos("DNI", 20, "DNI", X, y)
capturar_datos("DNI", 20, "DNI", X, y)

# Entrenamiento del modelo
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
print(" Modelo entrenado correctamente")

# Clasificación en tiempo real
cap = cv2.VideoCapture(0)
print("Clasificación en vivo. Presiona ESPACIO para predecir o ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    key = cv2.waitKey(1)

    if key == 32:  # ESPACIO para predecir
        feature = extraer_caracteristicas_avanzadas(frame).reshape(1, -1)
        pred = model.predict(feature)[0]
        print(f" Predicción: {pred}")
        cv2.putText(display, f"Es una {pred}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif key == 27:  # ESC para salir
        break

    cv2.imshow("Clasificador", display)

cap.release()
cv2.destroyAllWindows()
