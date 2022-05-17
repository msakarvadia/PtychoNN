import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
BATCH_SIZE = 32

data_dir = "/lus/theta-fs0/projects/datascience/mansisak/S26_beamtime"
concat = False
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename[-4:] == ".npz":
            print(os.path.join(root, filename))
            data_path = os.path.join(root, filename)
            
            #Condition check to see if the first scan has been loaded
            if 'real_space' in locals():
                concat = True
                old_real_space = real_space
                old_reciprocal = reciprocal
                
       
            real_space = np.load(data_path)['real']
            reciprocal = np.load(data_path)['reciprocal']
            
            #If we are loading scan # 2 or greater, we concatenate the scans
            if concat:
                real_space = np.concatenate((old_real_space, real_space))
                reciprocal = np.concatenate((old_reciprocal, reciprocal))

amp = np.abs(real_space)
ph = np.angle(real_space)

#Split test, train, val

X_train, remaining_data = np.split(reciprocal, [int(0.8 * len(reciprocal))])
X_val, X_test = np.split(remaining_data,[int(0.5 * len(remaining_data))])

Y_I_train, remaining_data = np.split(amp, [int(0.8 * len(amp))])
Y_I_val, Y_I_test = np.split(remaining_data,[int(0.5 * len(remaining_data))])

Y_phi_train, remaining_data = np.split(ph, [int(0.8 * len(ph))])
Y_phi_val, Y_phi_test = np.split(remaining_data,[int(0.5 * len(remaining_data))])

print("Train: ", X_train.shape, "Val: ", X_val.shape, "Test: ", X_test.shape)

#Add the color channel:
X_train = X_train[:,np.newaxis,:]
X_val = X_val[:,np.newaxis,:]
X_test = X_test[:,np.newaxis,:]

Y_I_train = Y_I_train[:,np.newaxis,:]
Y_I_val = Y_I_val[:,np.newaxis,:]
Y_I_test = Y_I_test[:,np.newaxis,:]

Y_phi_train = Y_phi_train[:,np.newaxis,:]
Y_phi_val = Y_phi_val[:,np.newaxis,:]
Y_phi_test = Y_phi_test[:,np.newaxis,:]

#Training data
X_train_tensor = torch.Tensor(X_train) 
Y_I_train_tensor = torch.Tensor(Y_I_train) 
Y_phi_train_tensor = torch.Tensor(Y_phi_train)

#Val data
X_val_tensor = torch.Tensor(X_val) 
Y_I_val_tensor = torch.Tensor(Y_I_val) 
Y_phi_val_tensor = torch.Tensor(Y_phi_val)

#Test data
X_test_tensor = torch.Tensor(X_test) 
Y_I_test_tensor = torch.Tensor(Y_I_test) 
Y_phi_test_tensor = torch.Tensor(Y_phi_test)

train_data = TensorDataset(X_train_tensor,Y_I_train_tensor,Y_phi_train_tensor)
test_data = TensorDataset(X_test_tensor,Y_I_test_tensor,Y_phi_test_tensor)
val_data = TensorDataset(X_val_tensor,Y_I_val_tensor,Y_phi_val_tensor)

#download and load training data
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

validloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

#same for test
#download and load training data
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

#Construct network
nconv = 32


class recon_model(nn.Module):

    def __init__(self):
        super(recon_model, self).__init__()


        self.encoder = nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
          nn.Conv2d(in_channels=1, out_channels=nconv, kernel_size=3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv, nconv, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.MaxPool2d((2,2)),

          nn.Conv2d(nconv, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),          
          nn.ReLU(),
          nn.MaxPool2d((2,2)),

          nn.Conv2d(nconv*2, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),          
          nn.ReLU(),
          nn.MaxPool2d((2,2)),
          )

        self.decoder1 = nn.Sequential(

          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),
            
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*2, 1, 3, stride=1, padding=(1,1)),
          nn.Sigmoid() #Amplitude model
          )

        self.decoder2 = nn.Sequential(

          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),
            
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*2, 1, 3, stride=1, padding=(1,1)),
          nn.Tanh() #Phase model
          )
    
    def forward(self,x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        #Restore -pi to pi range
        ph = ph*np.pi #Using tanh activation (-1 to 1) for phase so multiply by pi

        return amp,ph

#Sanity check to ensure dataloader is properly configured:
model = recon_model()
for ft_images,amps,phs in trainloader:
    print("batch size:", ft_images.shape)
    amp, ph = model(ft_images)
    print(amp.shape, ph.shape)
    print(amp.dtype, ph.dtype)
    break
