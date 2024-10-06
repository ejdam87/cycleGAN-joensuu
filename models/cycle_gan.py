import torch
import itertools

import gan_loss
import unet
import patch_gan

class CycleGAN(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:

        # ---
        # Mapping X -> Y
        self.G = unet.UNet(in_channels, out_channels)

        # Mapping Y -> X
        self.F = unet.UNet(out_channels, in_channels)

        # Classifier distinguishing real images from X and generated to X
        self.D_x = patch_gan.PatchGAN(in_channels)

        # Classifier distinguishing real images from Y and generated to Y
        self.D_y = patch_gan.PatchGAN(in_channels)
        # ---

        # --- Losses
        self.criterion_GAN = gan_loss.GANLoss()  # classic GAN
        self.criterionCycle = torch.nn.L1Loss()  # G( F(y) ) ≈ y, F( G(x) ) ≈ x
        self.criterionIdt = torch.nn.L1Loss()    # Color preservation
        # ---

        # --- Optimizers
        self.optimizer_GF = torch.optim.Adam(itertools.chain(self.G.parameters(), self.F.parameters()))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_x.parameters(), self.D_y.parameters()))
        # ---
