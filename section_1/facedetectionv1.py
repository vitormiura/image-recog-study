from sys import maxsize
import cv2

image = cv2.imread('section_1/pessoas.jpg')

classifier = cv2.CascadeClassifier('section_1/haarcascade_frontalface_default.xml') #Utiliza o xml como classificador
grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Converte a imagem para cinza

detection = classifier.detectMultiScale(grayImg, scaleFactor=1.09, minNeighbors=5, minSize=(30,30), maxSize=(40,40)) #fator de escala, vizinhos minimos e tamanho max/min de um obj.

#print(detection) 
print(len(detection)) #num de faces detectadas

for (x,y,l,a) in detection: #(left, top, width, height)
    cv2.rectangle(image, (x,y), (x + l, y + a), (0,255,0), 2)

cv2.imshow('Bounding box example', image)
cv2.waitKey(0)
cv2.destroyAllWindows()