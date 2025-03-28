import torch, torchvision
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
import numpy as np
from skimage.transform import resize

import os
from typing import Tuple

from sklearn.utils import shuffle

from transformers import ViTModel, ViTConfig

# define parameters
EPOCHS = 60
NGPUS = 1
BATCH_SIZE = NGPUS * 64
LR = NGPUS * 1e-3
print("GPUs:", NGPUS, "Batch size:", BATCH_SIZE, "Learning rate:", LR)

# x,y dimension size
H, W = 64, 64

# training, test, validation data size
NLINES = 100  #How many lines of data to use for training?
NLTEST = 60  #How many lines for the test set?

N_TRAIN = NLINES * 161
N_VALID = 805  #How much to reserve for validation

# define path for saved model
path = os.getcwd()

MODEL_SAVE_PATH = path + '/trained_model/'
if (not os.path.isdir(MODEL_SAVE_PATH)):
    os.mkdir(MODEL_SAVE_PATH)

# data path
data_path = path + '/../../data/20191008_39_diff.npz'
label_path = path + '/../../data/20191008_39_amp_pha_10nm_full.npy'

data_path='/gpfs/mira-home/mansisak/PtychoNN/data/20191008_39_diff.npz'

def prepare_dataloader(datapath,
                       label_path) -> Tuple[torch.utils.data.DataLoader]:
    # load data
    print(data_path)
    data_diffr = np.load(data_path, allow_pickle=True)['arr_0']
    real_space = np.load(label_path)
    amp = np.abs(real_space)
    ph = np.angle(real_space)
    print(amp.shape)
    print(data_diffr.shape)

    # crop diff to (64,64)
    data_diffr_red = np.zeros(
        (data_diffr.shape[0], data_diffr.shape[1], 64, 64), float)
    #for i in tqdm(range(data_diffr.shape[0])):
    for i in (range(data_diffr.shape[0])):
        for j in range(data_diffr.shape[1]):
            data_diffr_red[i, j] = resize(data_diffr[i, j, 32:-32, 32:-32],
                                          (64, 64),
                                          preserve_range=True,
                                          anti_aliasing=True)
            data_diffr_red[i, j] = np.where(data_diffr_red[i, j] < 3, 0,
                                            data_diffr_red[i, j])

    # split training and testing data
    tst_strt = amp.shape[0] - NLTEST  #Where to index from
    print(tst_strt)

    X_train = data_diffr_red[:NLINES, :].reshape(-1, H, W)[:, np.newaxis, :, :]
    Y_I_train = amp[:NLINES, :].reshape(-1, H, W)[:, np.newaxis, :, :]
    Y_phi_train = ph[:NLINES, :].reshape(-1, H, W)[:, np.newaxis, :, :]

    print(X_train.shape)

    X_train, Y_I_train, Y_phi_train = shuffle(X_train,
                                              Y_I_train,
                                              Y_phi_train,
                                              random_state=0)

    #Training data
    X_train_tensor = torch.Tensor(X_train)
    Y_I_train_tensor = torch.Tensor(Y_I_train)
    Y_phi_train_tensor = torch.Tensor(Y_phi_train)

    print(Y_phi_train.max(), Y_phi_train.min())

    print("x_train tensor, y_i train tensor, y_phi train tensor")
    print(X_train_tensor.shape, Y_I_train_tensor.shape,
          Y_phi_train_tensor.shape)

    train_data = TensorDataset(X_train_tensor, Y_I_train_tensor,
                               Y_phi_train_tensor)

    # split training and validation data
    train_data2, valid_data = torch.utils.data.random_split(
        train_data, [N_TRAIN - N_VALID, N_VALID])
    print(len(train_data2), len(valid_data))  #, len(test_data)

    #download and load training data
    trainloader = DataLoader(train_data2,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=4)
    validloader = DataLoader(valid_data,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=4)
    return trainloader, validloader

#define vit
configuration = ViTConfig(image_size = 64,
                          num_channels=1,
                          hidden_size = 256,
                          patch_size = 16,
                          num_attention_heads=8)
vit_I = ViTModel(configuration)
vit_phi = ViTModel(configuration)

# define network
nconv = 32
class recon_model(nn.Module):
    def __init__(self, vit_I, vit_phi):
        super(recon_model, self).__init__()

        self.vit_I = vit_I
        self.vit_phi = vit_phi

    def forward(self, x):
        amp = self.vit_I(x).last_hidden_state[:, 0:16, :][:, None, :, : ]
        ph = self.vit_phi(x).last_hidden_state[:, 0:16, :][:, None, :, : ]

        amp = torch.reshape(amp, (-1, 1, 64, 64))
        ph = torch.reshape(ph, (-1, 1, 64, 64))

        #Restore -pi to pi range
        ph = ph * np.pi  #Using tanh activation (-1 to 1) for phase so multiply by pi

        return amp, ph


# load data
trainloader, validloader = prepare_dataloader(data_path, label_path)


# check model
model = recon_model(vit_I, vit_phi)
for ft_images, amps, phs in trainloader:
    print("batch size:", ft_images.shape)
    amp, ph = model(ft_images)
    print(amp.shape, ph.shape)
    print(amp.dtype, ph.dtype)
    break

#summary(model, (1, H, W), device="cpu")

# move model to device
# use DataParallel if NGPUS larger than 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if NGPUS > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)  #Default all devices

model = model.to(device)

#Optimizer details
iterations_per_epoch = np.floor(
    (N_TRAIN - N_VALID) /
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

        pred_amps, pred_phs = model(ft_images)
        '''
        output_I = vit_I(ft_images)
        output_phi = vit_phi(ft_images)
        #print(output.last_hidden_state[:, 0:256, : ].shape)
        hidden_state_I = output_I.last_hidden_state[:, 0:16, :][:, None, :, : ]
        hidden_state_phi = output_phi.last_hidden_state[:, 0:16, :][:, None, :, : ]
        if hidden_state_I.shape != (64,1,16,256):
            continue
        #The last token in the sequence length dimension is the [CLS] token - we will ignore it for now
        #TODO look into why the dimensions arn't consistent
        pred_amps = torch.reshape(hidden_state_I, (64, 1, 64, 64))
        pred_phs = torch.reshape(hidden_state_phi, (64, 1, 64, 64))
        '''
    

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
        
        pred_amps, pred_phs = model(ft_images)
        """
        output_I = vit_I(ft_images)
        output_phi = vit_phi(ft_images)
        #print(output.last_hidden_state[:, 0:256, : ].shape)
        hidden_state_I = output_I.last_hidden_state[:, 0:16, :][:, None, :, : ]
        hidden_state_phi = output_phi.last_hidden_state[:, 0:16, :][:, None, :, : ]
        if hidden_state_I.shape != (64,1,16,256):
            continue
        #The last token in the sequence length dimension is the [CLS] token - we will ignore it for now
        #TODO look into why the dimensions arn't consistent
        pred_amps = torch.reshape(hidden_state_I, (64, 1, 64, 64))
        pred_phs = torch.reshape(hidden_state_phi, (64, 1, 64, 64))
        """

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

    X_test = np.load(path + '/../../data/X_test.npy')
    Y_I_test = np.load(path + '/../../data/Y_I_test.npy')
    Y_phi_test = np.load(path + '/../../data/Y_phi_test.npy')

    X_test = X_test.reshape(-1, H, W)[:, np.newaxis, :, :]
    Y_I_test = Y_I_test.reshape(-1, H, W)[:, np.newaxis, :, :]
    Y_phi_test = Y_phi_test.reshape(-1, H, W)[:, np.newaxis, :, :]

    print('test data shape:', X_test.shape, Y_I_test.shape, Y_phi_test.shape)

    #Test data
    #TODO include the ground truth data as well!! --> Look at validation and test loss
    X_test_tensor = torch.Tensor(X_test)
    Y_I_test_tensor = torch.Tensor(Y_I_test)
    Y_phi_test_tensor = torch.Tensor(Y_phi_test)

    test_data = TensorDataset(X_test_tensor, Y_I_test_tensor, Y_phi_test_tensor)
    testloader = DataLoader(test_data,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=4)

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

    # stitching
    point_size = 3
    overlap = 4 * point_size
    tst_side = 60

    composite_amp = np.zeros(
        (tst_side * point_size + overlap, tst_side * point_size + overlap),
        float)
    ctr = np.zeros_like(composite_amp)
    data_reshaped = amps.reshape(
        tst_side, tst_side, 64,
        64)[:, :, 32 - int(overlap / 2):32 + int(overlap / 2),
            32 - int(overlap / 2):32 + int(overlap / 2)]

    for i in range(tst_side):
        for j in range(tst_side):
            composite_amp[point_size * i:point_size * i + overlap, point_size *
                          j:point_size * j + overlap] += data_reshaped[i, j]
            ctr[point_size * i:point_size * i + overlap,
                point_size * j:point_size * j + overlap] += 1

    composite_phase = np.zeros(
        (tst_side * point_size + overlap, tst_side * point_size + overlap),
        float)
    ctr = np.zeros_like(composite_phase)
    data_reshaped = phs.reshape(
        tst_side, tst_side, 64,
        64)[:, :, 32 - int(overlap / 2):32 + int(overlap / 2),
            32 - int(overlap / 2):32 + int(overlap / 2)]

    for i in range(tst_side):
        for j in range(tst_side):
            composite_phase[point_size * i:point_size * i + overlap,
                            point_size * j:point_size * j +
                            overlap] += data_reshaped[i, j]
            ctr[point_size * i:point_size * i + overlap,
                point_size * j:point_size * j + overlap] += 1

    stitched_phase = composite_phase[int(overlap / 2):-int(overlap / 2),
                                     int(overlap / 2):-int(overlap / 2)] / ctr[
                                         int(overlap / 2):-int(overlap / 2),
                                         int(overlap / 2):-int(overlap / 2)]

    stitched_amp = composite_amp[int(overlap / 2):-int(overlap / 2),
                                 int(overlap / 2):-int(overlap / 2)] / ctr[
                                     int(overlap / 2):-int(overlap / 2),
                                     int(overlap / 2):-int(overlap / 2)]

    stitched_amp_down = resize(stitched_amp, (60, 60),
                               preserve_range=True,
                               anti_aliasing=True)
    stitched_phase_down = resize(stitched_phase, (60, 60),
                                 preserve_range=True,
                                 anti_aliasing=True)

    # ground truth amplitude and phase
    true_amp = Y_I_test.reshape(NLTEST, NLTEST, 64, 64)
    true_ph = Y_phi_test.reshape(NLTEST, NLTEST, 64, 64)

    return true_amp, true_ph, stitched_amp_down, stitched_phase_down



# start training
metrics = {'losses': [], 'val_losses': [], 'lrs': [], 'best_val_loss': np.inf, 'test_losses' : []}
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
true_amp, true_ph, stitched_amp_down, stitched_phase_down = test(model)
print('FT  | Test Loss: %.5f ' %
      (metrics['test_losses'][-1][0]))
print('Amp | Test Loss: %.4f ' %
      (metrics['test_losses'][-1][1]))
print('Ph  | Test Loss: %.3f ' %
      (metrics['test_losses'][-1][2]))
print("FINISHED TESTING")
