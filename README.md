# explicacion, deteccion de ojos.

1: importamos la libreias  

import cv2
from datetime import datetime


2: cargamos los modelos 

face_cascade = cv2.CascadeClassifier(...)
eye_cascade = cv2.CascadeClassifier(...)

3: inicamosmla camra con     (cap = cv2.VideoCapture(0))

4: creamos el buqle que capturara  la convercion, detectara los rostros,  buscara los "ojos dentroo" identificara los hojos y  le pondra circulos en los ojos.

5: en la terminal saldra lafehca y hora de cada decto de hojo y tambien saldra en la venta que desplegara pyhon al detectar los ojos 


6: para salir de vnetna que nos dara python con la camra prendica hay que precioner control x, si es linux.


7: en conclusion es codgio es un progrma parte del entregable para decttectar ojos en tiempo real 
