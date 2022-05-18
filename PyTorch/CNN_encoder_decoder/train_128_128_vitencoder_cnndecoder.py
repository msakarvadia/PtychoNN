import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import ViTModel, ViTConfig
from torch.utils.data import TensorDataset, DataLoader

# define parameters
EPOCHS = 60
NGPUS = 1
BATCH_SIZE = NGPUS * 32
LR = NGPUS * 1e-3
print("GPUs:", NGPUS, "Batch size:", BATCH_SIZE, "Learning rate:", LR)                       

# define path for saved model
path = os.getcwd()

MODEL_SAVE_PATH = path + '/trained_model_128_128/'
if (not os.path.isdir(MODEL_SAVE_PATH)):
    os.mkdir(MODEL_SAVE_PATH)                         

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

#define vit
configuration = ViTConfig(image_size = 128,
                          num_channels=1,
                          hidden_size = 1024,
                          patch_size = 16,
                          num_attention_heads=8)
vit = ViTModel(configuration)

# define network
nconv = 32

class recon_model(nn.Module):
    def __init__(self, vit):
        super(recon_model, self).__init__()

        self.vit = vit

        self.encoder = nn.Sequential(  # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            #nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=4),
            nn.Conv2d(in_channels=1,
                      out_channels=nconv,
                      kernel_size=3,
                      stride=1,
                      padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv, nconv, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(nconv, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(nconv * 2, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nconv * 4, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nconv * 2, 1, 3, stride=1, padding=(1, 1)),
            nn.Sigmoid()  #Amplitude model
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nconv * 4, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(nconv * 2, 1, 3, stride=1, padding=(1, 1)),
            nn.Tanh()  #Phase model
        )

    def forward(self, x):
        x = self.vit(x).last_hidden_state[:, 0:16, :][:, None, :, : ]
        x = torch.reshape(x, (-1, 1, 128, 128))

        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        #Restore -pi to pi range
        ph = ph * np.pi  #Using tanh activation (-1 to 1) for phase so multiply by pi

        return amp, ph

#Sanity check to ensure dataloader is properly configured:
model = recon_model(vit)
for ft_images,amps,phs in trainloader:
    print("batch size:", ft_images.shape)
    amp, ph = model(ft_images)
    print(amp.shape, ph.shape)
    print(amp.dtype, ph.dtype)
    break

# DataParallel if NGPUS larger than 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if NGPUS > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)  #Default all devices

model = model.to(device)

#Optimizer details
iterations_per_epoch = np.floor(
    (X_train.shape[0]) /
    BATCH_SIZE) + 1  #Final batch will be less than batch size
step_size = 6 * iterations_per_epoch  #Paper recommends 2-10 number of iterations, step_size is half cycle
print("LR step size is:", step_size,
      "which is every %d epochs" % (step_size / iterations_per_epoch))

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                              base_lr=LR / 10,
                                              max_lr=LR,
                                              step_size_up=step_size,
                                              cycle_momentum=False,
                                              mode='triangular2')


#Function to update saved model if validation loss is minimum
def update_saved_model(model, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
    if (NGPUS > 1):
        torch.save(
            model.module.state_dict(), path + 'best_model.pth'
        )  #Have to save the underlying model else will always need 4 GPUs
    else:
        torch.save(model, path + 'best_model.pth')


# define training
def train(trainloader, metrics):
    tot_loss = 0.0
    loss_amp = 0.0
    loss_ph = 0.0

    for i, (ft_images, amps, phs) in tqdm(enumerate(trainloader)):
        ft_images = ft_images.to(device)  #Move everything to device
        amps = amps.to(device)
        phs = phs.to(device)

        #The last token in the sequence length dimension is the [CLS] token - we will ignore it for now
    
        pred_amps, pred_phs = model(ft_images)  #Forward pass

        #Compute losses
        loss_a = criterion(pred_amps, amps)  #Monitor amplitude loss
        loss_p = criterion(pred_phs, phs)  #Monitor phase loss
        loss = loss_a + loss_p  #Use equiweighted amps and phase

        #Zero current grads and do backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.detach().item()
        loss_amp += loss_a.detach().item()
        loss_ph += loss_p.detach().item()

        #Update the LR according to the schedule -- CyclicLR updates each batch
        scheduler.step()
        metrics['lrs'].append(scheduler.get_last_lr())

    #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
    metrics['losses'].append([tot_loss / i, loss_amp / i, loss_ph / i])


# define validation
def validate(validloader, metrics):
    tot_val_loss = 0.0
    val_loss_amp = 0.0
    val_loss_ph = 0.0
    for j, (ft_images, amps, phs) in enumerate(validloader):
        ft_images = ft_images.to(device)
        amps = amps.to(device)
        phs = phs.to(device)
        
        pred_amps, pred_phs = model(ft_images)  #Forward pass

        val_loss_a = criterion(pred_amps, amps)
        val_loss_p = criterion(pred_phs, phs)
        val_loss = val_loss_a + val_loss_p

        tot_val_loss += val_loss.detach().item()
        val_loss_amp += val_loss_a.detach().item()
        val_loss_ph += val_loss_p.detach().item()
    metrics['val_losses'].append(
        [tot_val_loss / j, val_loss_amp / j, val_loss_ph / j])

    #Update saved model if val loss is lower
    if (tot_val_loss / j < metrics['best_val_loss']):
        print(
            "Saving improved model after Val Loss improved from %.5f to %.5f" %
            (metrics['best_val_loss'], tot_val_loss / j))
        metrics['best_val_loss'] = tot_val_loss / j
        update_saved_model(model, MODEL_SAVE_PATH)


# define test
def test(model):
    tot_test_loss = 0.0
    test_loss_amp = 0.0
    test_loss_ph = 0.0

    model.eval()  #imp when have dropout etc
    amps = []
    phs = []

    for i, (ft_images, amplitude, phase) in enumerate(testloader):
        #####
        ft_images = ft_images.to(device)
        ground_truth_amps = amplitude.to(device)
        ground_truth_phs = phase.to(device)
        
        pred_amps, pred_phs = model(ft_images)

        test_loss_a = criterion(pred_amps, ground_truth_amps)
        test_loss_p = criterion(pred_phs, ground_truth_phs)
        test_loss = test_loss_a + test_loss_p

        tot_test_loss += test_loss.detach().item()
        test_loss_amp += test_loss_a.detach().item()
        test_loss_ph += test_loss_p.detach().item()
        #####

        #ft_images = ft_images[0].to(device)
        amp, ph = model(ft_images)
        for j in range(ft_images.shape[0]):
            amps.append(amp[j].detach().to("cpu").numpy())
            phs.append(ph[j].detach().to("cpu").numpy())

    metrics['test_losses'].append([tot_test_loss / (i), test_loss_amp / (i), test_loss_ph / (i)])

    amps = np.array(amps).squeeze()
    phs = np.array(phs).squeeze()
    print('test output amp shape and dtype:', amps.shape, amps.dtype)
    print('test output phase shape and dtype:', phs.shape, phs.dtype)




# start training
metrics = {'losses': [], 'val_losses': [], 'lrs': [], 'test_losses':[], 'best_val_loss': np.inf}
for epoch in range(EPOCHS):

    # Set model to train mode
    model.train()

    # Training loop
    train(trainloader, metrics)

    # Switch model to eval mode
    model.eval()

    # Validation loop
    validate(validloader, metrics)

    print('Epoch: %d | FT  | Train Loss: %.5f | Val Loss: %.5f' %
          (epoch, metrics['losses'][-1][0], metrics['val_losses'][-1][0]))
    print('Epoch: %d | Amp | Train Loss: %.4f | Val Loss: %.4f' %
          (epoch, metrics['losses'][-1][1], metrics['val_losses'][-1][1]))
    print('Epoch: %d | Ph  | Train Loss: %.3f | Val Loss: %.3f' %
          (epoch, metrics['losses'][-1][2], metrics['val_losses'][-1][2]))
    print('Epoch: %d | Ending LR: %.6f ' % (epoch, metrics['lrs'][-1][0]))

# testing
test(model)
print('FT  | Test Loss: %.5f ' %
      (metrics['test_losses'][-1][0]))
print('Amp | Test Loss: %.4f ' %
      (metrics['test_losses'][-1][1]))
print('Ph  | Test Loss: %.3f ' %
      (metrics['test_losses'][-1][2]))
print("FINISHED TESTING")
