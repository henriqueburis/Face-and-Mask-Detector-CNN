import cv2
import torch
import torchvision
import torch.nn as nn
import numpy as np
import imutils
import tensorflow as tf
from torchvision import transforms
#from keras.preprocessing.image import img_to_array
#from google.colab.patches import cv2_imshow


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

print('==> Building model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device) #CPU ou Cuda
net = torchvision.models.resnet50(pretrained=True)
#net = torchvision.models.vgg19(pretrained=True)
#net.classifier[6] = nn.Linear(4096,2) # dois rotulos

modules = list(net.children())[:-1]
modules.append(nn.Flatten())
modules.append(nn.Linear(net.fc.in_features,2))
net = nn.Sequential(*modules)
net = net.to(device)

model = net.to(device)
print('==> load model..')
model.load_state_dict(torch.load("results/CNN_face_vgg19_face_model.pt",map_location=device)) # carregar o modelo treinado "CNN"
print('==> load detector..')
detector = cv2.CascadeClassifier("Data_haarcascade/haarcascade_frontalface_default.xml")

print('==> load video or cam..')
#camera = cv2.VideoCapture(0) # usar uma camemra
camera = cv2.VideoCapture("23245978_144521.mp4") # carregar o video

model = model.eval()
while True:
  (grabbed, frame) = camera.read()
  
  frame = imutils.resize(frame, width=450) # se nao tiver o arquivo vai dar erro aqui
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frameClone = frame.copy()

  rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, 
                                    minSize=(32, 32), flags=cv2.CASCADE_SCALE_IMAGE)
  
  for (fX, fY, fW, fH) in rects:
    
    roi = gray[fY:fY + fH, fX:fX + fW]
    cv2.imshow('roi', cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    tensor_roi = transform(roi) # transformar image recortada em tensor para o modelo inferir.

    outputs = model(tensor_roi.unsqueeze(0)) # predizer a img
    _, predicted = outputs.max(1)

    if predicted.detach().numpy() == [0] :
        cv2.putText(frameClone, "Protegido", (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
    else:
         cv2.putText(frameClone, "Desprotegido", (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
         cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2) 


  cv2.imshow('Face', frameClone)
  #cv2_imshow(frameClone) # esta linha so é utilizada caso use o googlecolab
  

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

camera.release()
cv2.destroyAllWindows()
