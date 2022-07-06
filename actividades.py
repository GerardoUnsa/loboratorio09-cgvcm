##### PREGUNTA 1 #####
'''
import cv2
import numpy as np

img = cv2.imread('messi5.jpg')

# Convertir la imgen de BGR (Blue, Green, Red) to HSV (Hue, Saturation, Value)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

#Se define el rango de colores del azul, se utiliza 
#para una mascara de objetos de color azul en la imagen

mask = cv2.inRange(hsv, lower_blue, upper_blue)


#Solo los colores del rango se resaltaran

res = cv2.bitwise_and(img, img, mask = mask)

cv2.imshow('Imagen',img)
cv2.imshow('Mascara',mask)
cv2.imshow('Resaltar azul',res)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

##### PREGUNTA 2 #####
'''
import cv2
import numpy as np

img = cv2.imread('messi5.jpg') # Capturar datos de imagen

# Cambiar de color BGR a escala de grises GRAY
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Mostrar en pantalla
cv2.imshow('Escala de grises', imgray)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

##### PREGUNTA 3 #####
'''
import cv2
import numpy as np

img = cv2.imread('messi5.jpg')

# Convertir la imgen de BGR (Blue, Green, Red) a HSV (Hue, Saturation, Value)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_green = np.array([30,50,50])
upper_green = np.array([80,255,255])

lower_red = np.array([160,50,50])
upper_red = np.array([220,255,255])

#Se define el rango de colores del rojo y verde, se utiliza 
#para una mascara de objetos de color rojo y verde en la imagen
mask = cv2.inRange(hsv, lower_green, upper_green)
mask2 = cv2.inRange(hsv, lower_red, upper_red)

#Solo los colores del rango se resaltaran
res = cv2.bitwise_and(img, img, mask = mask)
res2 = cv2.bitwise_and(img, img, mask = mask2)
res3 = cv2.bitwise_or(res, res2) # Agrupar rojo y verde

cv2.imwrite('messi_red_green.jpg',res3) # Guardar imagen (direccion+nombre, imagen)
cv2.imshow('Imagen',img)
cv2.imshow('Resaltar ambos',res3)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

##### PREGUNTA 4 y 5 #####
'''
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('messi5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces:
	#cv2.rectangle(img, coordenada, radio, color, grosor de linea)
	cv2.circle(img, (x,y), 40, (255,0,0), 3)
	cv2.putText(img, "Persona", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255) )

cv2.imwrite('rostros_messi.jpg', img) # Guardar imagen (direccion+nombre, imagen)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

##### PREGUNTA 6 #####
'''
import cv2
import numpy as np

img = cv2.imread('messi5.jpg')
_, frame = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
_, frame2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
_, frame3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
_, frame4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
_, frame5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

cv2.imshow('Image', img)
cv2.imshow('binary', frame)
cv2.imshow('binary_inv', frame2)
cv2.imshow('trunc', frame3)
cv2.imshow('tozero', frame4)
cv2.imshow('tozero_inv', frame5)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

##### PREGUNTA 7 #####

import cv2
import numpy as np

img = cv2.imread('messi5.jpg')
cv2.imshow('image', img)

def click_event(event, x, y, flags, param):
	print(x,y)
	if event == cv2.EVENT_LBUTTONDOWN: # Click izquierdo
		cv2.rectangle(img, (x,y),(x+120,y+60),(0,0,255),2) # Rectangulo
		cv2.imshow('image', img)

	if event == cv2.EVENT_RBUTTONUP: # CLick derecho
		cv2.circle(img, (x,y), 30, (0,255,0),-1) # Circulo, 30 radio
		cv2.imshow('image', img)

	cv2.imwrite('mouse_events_messi.jpg', img) # Guardar imagen

cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()