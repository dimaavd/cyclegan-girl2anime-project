import torch
import torchvision.transforms as transforms

import itertools
from pathlib import Path
import torch.nn as nn

from Girl2animeDataset import Girl2animeDataset
from models import Generator,Discriminator,weights_init

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from train import fit

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

root = 'girl2anime'
num_workers = 6
batch_size = 5
epochs = 100
lr = 2*1e-4

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


dataset = Girl2animeDataset(root,transform, mode="train")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,num_workers =num_workers , shuffle=True, pin_memory=True)

model = {
    "G_AB": Generator().to(device),
    "D_A": Discriminator().to(device),
    "F_BA": Generator().to(device),
    "D_B": Discriminator().to(device),
}

model["G_AB"].apply(weights_init)
model["D_A"].apply(weights_init)
model["F_BA"].apply(weights_init)
model["D_B"].apply(weights_init)


optim_G_AB_and_F_BA = torch.optim.Adam(itertools.chain(model["G_AB"].parameters(), model["F_BA"].parameters()), lr=lr, betas=(0.5, 0.999))
optim_D_A_and_D_B = torch.optim.Adam(itertools.chain(model["D_A"].parameters(), model["D_B"].parameters()), lr=lr, betas=(0.5, 0.999))

optimizer = {
        "G_AB_and_F_BA": optim_G_AB_and_F_BA,
        "D_A_and_D_B": optim_D_A_and_D_B
    }

# "We keep the same learning rate for the first 100
# epochs and linearly decay the rate to zero over the next 100
# epochs."
def get_lr(epoch):
    start_decay_epoch = epochs // 2
    if epoch <= start_decay_epoch:
        return 1.0
    else:
        new_lr = (1 - (epoch - start_decay_epoch) / (start_decay_epoch))
        return new_lr

lr_scheduler_G_AB_and_F_BA = torch.optim.lr_scheduler.LambdaLR(optimizer["G_AB_and_F_BA"], lr_lambda=get_lr)
lr_scheduler_D_A_and_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer["D_A_and_D_B"], lr_lambda=get_lr)

scheduler = {
    "G_AB_and_F_BA": lr_scheduler_G_AB_and_F_BA,
    "D_A_and_D_B": lr_scheduler_D_A_and_D_B
}

criterion = {
    "MSE": nn.MSELoss(),
    "L1": nn.L1Loss()
}

results_path = Path("results")
plots_path = Path("plots")
weights_path = Path("weights")
results_path.mkdir(parents=True, exist_ok=True)
plots_path.mkdir(parents=True, exist_ok=True)
weights_path.mkdir(parents=True, exist_ok=True)

history = fit(model,optimizer,scheduler,criterion,dataloader,epochs,show_mode=False)