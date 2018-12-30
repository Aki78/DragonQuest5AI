import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import serial
import time
ImageFile.LOAD_TRUNCATED_IMAGES = True

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


from torch.autograd import Variable
import torch.nn.functional as F

class SimpleCNN(torch.nn.Module):
    # Shape = (3,32,32)
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(3,18,kernel_size=3,stride=1,padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = torch.nn.Linear(18*50//2*50//2, 64) #
        self.fc2 = torch.nn.Linear(64,2) # 2 different classes
        
    def forward(self, x):

        x = F.relu(self.conv1(x)) 

        x = self.pool(x)

        x = x.view(-1, 18*50*50//4) # 480

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return(x)
        

def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding))/stride) + 1
    return output


#MAIN START
CNN = SimpleCNN()
AModel = torch.load("DQClassifier50x50New.py")
cap = cv2.VideoCapture(0)
m = torch.nn.Upsample(size=(50,50))
#m = torch.nn.functional.interpolate(size=(50,50))

i = 0
step = 0
while 1:
    print('step=',step)
    step += 1
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     
    frame = torchvision.transforms.ToTensor()(frame)
    myInp = m(torch.unsqueeze(frame,0))
    print(AModel.forward(myInp).argmax().item())
    val = AModel.forward(myInp).argmax()

    print(val.numpy())
    if val.numpy()== 0:

        os.system("echo \"b\" > /dev/ttyACM0")

        time.sleep(0.35)
    else:
        os.system("echo \"a\" > /dev/ttyACM0")

        time.sleep(0.35)
        
    print("EndStep")
    if ret == True:
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    else:
        break
        
cap.release()
cv2.destroyAllWindows()
