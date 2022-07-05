import torch
#import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from time import time
from pathlib import Path


from save_history_and_results import plot_history,show_samples
from IPython.display import clear_output


class Buffer():
    def __init__(self,buffer_size):
        self.buffer_size=buffer_size
        self.buffer_images = []
    def get_images(self,fake_images):
        output = []
        for fake_image in fake_images:
            fake_image = torch.unsqueeze(fake_image.detach(), 0)
            if len(self.buffer_images)<self.buffer_size:
                # fill the buffer first time
                output.append(fake_image)
                self.buffer_images.append(fake_image)
            else:
                if np.random.uniform(0,1) > 0.5:
                    # return random element from buffer and
                    # replace it with a new image
                    index = np.random.randint(0,self.buffer_size)
                    output.append(self.buffer_images[index])
                    self.buffer_images[index]=fake_image
                else:
                    # return new image
                    output.append(fake_image)
        return torch.cat(output).to('cuda')


def fit(model, optimizer, scheduler, criterion, dataloader, epochs, show_mode=False):
    losses_G_AB = []
    losses_F_BA = []
    forward_cycle_consistency_losses = []
    backward_cycle_consistency_losses = []
    identity_losses_G_AB = []
    identity_losses_F_BA = []
    losses_D_A = []
    losses_D_B = []

    time_epochs = []

    batch_num = len(dataloader)
    # "We keep an image buffer that stores the 50 previously created images."
    buffer_A = Buffer(50)
    buffer_B = Buffer(50)
    for epoch in range(epochs):
        tic = time()
        loss_G_AB_per_epoch = []
        loss_F_BA_per_epoch = []
        forward_cycle_consistency_loss_per_epoch = []
        backward_cycle_consistency_loss_per_epoch = []
        identity_loss_G_AB_per_epoch = []
        identity_loss_F_BA_per_epoch = []
        loss_D_A_per_epoch = []
        loss_D_B_per_epoch = []
        for iteration, (real_images_A, real_images_B) in enumerate(dataloader):

            real_images_A, real_images_B = real_images_A.to('cuda'), real_images_B.to('cuda')
            #### train generators
            optimizer["G_AB_and_F_BA"].zero_grad()
            real_targets = torch.ones(real_images_A.size(0), 1, device='cuda')
            fake_targets = torch.zeros(real_images_A.size(0), 1, device='cuda')

            # Generator Adversarial Loss
            fake_images_B = model["G_AB"](real_images_A)
            fake_images_A = model["F_BA"](real_images_B)

            preds_A = model["D_A"](fake_images_A)
            preds_B = model["D_B"](fake_images_B)

            loss_G_AB = criterion["MSE"](preds_B, real_targets)
            loss_F_BA = criterion["MSE"](preds_A, real_targets)

            generator_adversarial_loss = loss_G_AB + loss_F_BA

            # Cycle Consistency Loss
            reconstructed_images_B = model["G_AB"](fake_images_A)
            reconstructed_images_A = model["F_BA"](fake_images_B)

            forward_cycle_consistency_loss = criterion["L1"](real_images_A, reconstructed_images_A)
            backward_cycle_consistency_loss = criterion["L1"](real_images_B, reconstructed_images_B)

            cycle_consistency_loss = forward_cycle_consistency_loss + backward_cycle_consistency_loss

            # identity mapping loss

            identity_images_B = model["G_AB"](real_images_B)
            identity_images_A = model["F_BA"](real_images_A)

            identity_loss_G_AB = criterion["L1"](identity_images_B, real_images_B)
            identity_loss_F_BA = criterion["L1"](identity_images_A, real_images_A)

            identity_mapping_loss = identity_loss_G_AB + identity_loss_F_BA

            # Full Objective

            # "For all the experiments, we set λ = 10 in Equation 3."
            lambda_coef = 10
            generator_full_loss = generator_adversarial_loss + lambda_coef*cycle_consistency_loss + identity_mapping_loss

            # Update generators weights
            generator_full_loss.backward()
            optimizer["G_AB_and_F_BA"].step()

            #### train discriminators

            optimizer["D_A_and_D_B"].zero_grad()

            # Discriminator Adversarial Loss
            real_preds_A = model["D_A"](real_images_A)
            real_preds_B = model["D_B"](real_images_B)

            real_loss_D_A = criterion["MSE"](real_preds_A, real_targets)
            real_loss_D_B = criterion["MSE"](real_preds_B, real_targets)

            buffer_fake_images_A = buffer_A.get_images(fake_images_A)
            buffer_fake_images_B = buffer_B.get_images(fake_images_B)

            fake_preds_A = model["D_A"](buffer_fake_images_A)
            fake_preds_B = model["D_B"](buffer_fake_images_B)

            fake_loss_D_A = criterion["MSE"](fake_preds_A, fake_targets)
            fake_loss_D_B = criterion["MSE"](fake_preds_B, fake_targets)

            loss_D_A = real_loss_D_A + fake_loss_D_A
            loss_D_B = real_loss_D_B + fake_loss_D_B

            discriminator_full_loss = loss_D_A + loss_D_B

            # Update discriminators weights
            discriminator_full_loss.backward()
            optimizer["D_A_and_D_B"].step()

            torch.cuda.empty_cache()


            loss_G_AB_per_epoch.append(loss_G_AB.item())
            loss_F_BA_per_epoch.append(loss_F_BA.item())

            forward_cycle_consistency_loss_per_epoch.append(forward_cycle_consistency_loss.item())
            backward_cycle_consistency_loss_per_epoch.append(backward_cycle_consistency_loss.item())

            identity_loss_G_AB_per_epoch.append(identity_loss_G_AB.item())
            identity_loss_F_BA_per_epoch.append(identity_loss_F_BA.item())

            loss_D_A_per_epoch.append(loss_D_A.item())
            loss_D_B_per_epoch.append(loss_D_B.item())
            if iteration == (batch_num - 1):
                clear_output(wait=True)
                show_samples(real_images_A, fake_images_B, real_images_B, fake_images_A, epoch, show_mode)

        scheduler["G_AB_and_F_BA"].step()
        scheduler["D_A_and_D_B"].step()
        # save model weights
        if ((epoch + 1) % (epochs//2)) == 0:
            torch.save(model["G_AB"].state_dict(), f"weights/G_AB_epoch_{epoch}.pth")
            torch.save(model["F_BA"].state_dict(), f"weights/F_BA_epoch_{epoch}.pth")
            torch.save(model["D_A"].state_dict(), f"weights/D_A_epoch_{epoch}.pth")
            torch.save(model["D_B"].state_dict(), f"weights/D_B_epoch_{epoch}.pth")
        toc = time()
        losses_G_AB.append(np.mean(loss_G_AB_per_epoch))
        losses_F_BA.append(np.mean(loss_F_BA_per_epoch))
        forward_cycle_consistency_losses.append(np.mean(forward_cycle_consistency_loss_per_epoch))
        backward_cycle_consistency_losses.append(np.mean(backward_cycle_consistency_loss_per_epoch))
        identity_losses_G_AB.append(np.mean(identity_loss_G_AB_per_epoch))
        identity_losses_F_BA.append(np.mean(identity_loss_F_BA_per_epoch))
        losses_D_A.append(np.mean(loss_D_A_per_epoch))
        losses_D_B.append(np.mean(loss_D_B_per_epoch))

        time_epochs.append(toc - tic)

        print(f"[{epoch + 1}/{epochs}]  time_epoch: {time_epochs[-1]:.2f}")

        history = [losses_G_AB,
                   losses_D_B,
                   losses_F_BA,
                   losses_D_A,
                   forward_cycle_consistency_losses,
                   backward_cycle_consistency_losses,
                   identity_losses_G_AB,
                   identity_losses_F_BA,
                   time_epochs]
        plot_history(history, show_mode)
    return history







