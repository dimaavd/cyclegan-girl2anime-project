import matplotlib.pyplot as plt


def plot_history(history, show_mode):
    losses_G_AB, losses_D_B, losses_F_BA, losses_D_A, forward_cycle_consistency_losses, backward_cycle_consistency_losses, identity_losses_G_AB, identity_losses_F_BA, time_epochs = history
    plt.figure(figsize=(15, 6))
    plt.plot(losses_G_AB, '-')
    plt.plot(losses_D_B, '-')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss_G_AB', 'loss_D_B'])
    plt.title('G_AB and D_B training losses')
    plt.savefig('./plots/G_AB_D_B_losses.png')
    if show_mode:
        plt.show()

    plt.figure(figsize=(15, 6))
    plt.plot(losses_F_BA, '-')
    plt.plot(losses_D_A, '-')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss_F_BA', 'loss_D_A'])
    plt.title('F_BA and D_A training losses')
    plt.savefig('./plots/F_BA_D_A_losses.png')
    if show_mode:
        plt.show()

    plt.figure(figsize=(15, 6))
    plt.plot(forward_cycle_consistency_losses, '-')
    plt.plot(backward_cycle_consistency_losses, '-')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['forward_cycle_consistency_loss', 'backward_cycle_consistency_loss'])
    plt.title('cycle consistency losses')
    plt.savefig('./plots/cycle_consistency_losses.png')
    if show_mode:
        plt.show()

    plt.figure(figsize=(15, 6))
    plt.plot(identity_losses_G_AB, '-')
    plt.plot(identity_losses_F_BA, '-')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['identity_loss_G_AB', 'identity_loss_F_BA'])
    plt.title('identity mapping losses')
    plt.savefig('./plots/identity_mapping_losses.png')
    if show_mode:
        plt.show()

    plt.figure(figsize=(15, 6))
    plt.plot(time_epochs, '-')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('time')
    plt.title('time_epochs')
    plt.savefig('./plots/time_epochs.png')
    if show_mode:
        plt.show()
    plt.close('all')

import torch
from torchvision.utils import make_grid


def show_samples(real_images_A, fake_images_B, real_images_B, fake_images_A, epoch, show_mode):
    batch_size = real_images_A.size(0)

    real_images_B = 0.5 * (real_images_B.detach() + 1.0)
    real_images_A = 0.5 * (real_images_A.detach() + 1.0)

    fake_images_B = 0.5 * (fake_images_B.detach() + 1.0)
    fake_images_A = 0.5 * (fake_images_A.detach() + 1.0)

    joint_batch = torch.cat([real_images_A, fake_images_B, real_images_B, fake_images_A]).cpu()

    fig = plt.figure(figsize=(10 * batch_size / 4, 10), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(make_grid(joint_batch, nrow=batch_size).permute(1, 2, 0), aspect='auto')
    plt.savefig('./results/result_epoch_{0:0=4d}.png'.format(epoch))
    if show_mode:
        plt.show()
    plt.close('all')