from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import numpy as np
# def transform():
#     return A.Compose(
#         [
#             A.Resize(height=32, width=128,interpolation=cv2.INTER_AREA , p=1.0),
#             ToTensor(),
#         ]
#     )   
class Predictor_image(nn.Module):
  def __init__(self,path_model="weights/cls_invoice.pth"):
    super().__init__()
    self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
    self.model.load_state_dict(torch.load(path_model))

  def process_input(self,img):

    img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
    return torch.FloatTensor(img).permute(2,0,1).unsqueeze(0)
  def forward(self,img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = self.process_input(img)

    result = self.model(img)
    result = torch.softmax(result,-1)
    return torch.argmax(result).item()

