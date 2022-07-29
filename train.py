# Train the model here
import torch.nn as nn
import torch
# from model import Net 
import numpy as np
import torch.nn.functional as F
import glob
import PIL.Image as pil_image
import h5py
# from model import Net
from model1 import Net
# from model2 import Net

model = Net()
# FILE = "nn-model-1.pth"
FILE = "nn-model-v2-2.pth"
model.load_state_dict(torch.load(FILE))
model.eval()

#Define hyper-parameters
num_epochs = 10
learning_rate = 1e-4
# --patch_size = 17
# --stride = 8

#Initialize loss function and optimizer
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#List of h5 files containing the patches
file_paths = list()
for i in range(13):
    fileName = "training-imgs/h5-files/file"+str(i)+".h5"
    file_paths.append(fileName)


#Train the model
for epoch in range(num_epochs):
    fileNum = 0
    for file_path in file_paths:
        file = h5py.File(file_path,'r')
        lr_patches = file.get('lr')
        hr_patches = file.get('hr')

        lr_patches = torch.tensor(np.array(lr_patches))
        hr_patches = torch.tensor(np.array(hr_patches))

        if(fileNum==0 and epoch==0): print("Starting training")
        output_patches = model(lr_patches)

        if(fileNum==0 and epoch==0): print("Model works")
        (N,n,m) = hr_patches.shape
        output_patches = output_patches.view(N,n,m)

        hr_patches = torch.tensor(hr_patches,dtype=torch.float)
        cost = loss(output_patches,hr_patches)
        if(fileNum==0 and epoch==0): print("Loss calculation successful")

        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

        statement = "Epoch: " + str(epoch) + ", filenum: " + str(fileNum+1) + ", loss: " + str(cost.item())
        print(statement)
        fileNum+=1

# FILE = "nn-model-v3.pth"
# torch.save(model.state_dict(),FILE)
