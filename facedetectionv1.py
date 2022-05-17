import cv2

image = cv2.imread('pessoas.jpg')

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Utiliza o xml como classificador
grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Converte a imagem para cinza

detection = classifier.detectMultiScale(grayImg) #Detecta faces e salva em matriz (left, top, width, height)

#print(detection) 
print(len(detection)) #num de faces detectadas

for (x,y,l,a) in detection:
    cv2.rectangle(image, (x,y), (x + l, y + a), (0,255,0), 2)

cv2.imshow('Bounding box example', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Haarscascade detectando falso positivo ainda.