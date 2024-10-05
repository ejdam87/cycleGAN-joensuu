import torch

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

        # --- losses
        self.criterion_GAN = gan_loss.GANLoss()
        self.criterionCylce = torch.nn.L1Loss()

        # TODO: what is the last loss for ???
        # ---
