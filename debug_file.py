# Core Imports
import time
import random

# External Dependency Imports
from imageio import imwrite
import torch
import numpy as np

# Internal Project Imports
from pretrained.vgg import Vgg16Pretrained
from utils import misc as misc
from utils.misc import load_path_for_pytorch
from utils.stylize import produce_stylization

# Fix Random Seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# Interpret command line arguments
content_path = './inputs/content/C2.png'
style_path = './inputs/style/S3.jpg'
output_path = './output.jpg'
flip_aug = False
max_scls = 4
sz = 512
content_loss = True
misc.USE_GPU = True
dont_colorize = True
content_weight = 0.25

# Error checking for arguments error checking for paths deferred to imageio
assert (0.0 <= content_weight) and (content_weight <= 1.0), "alpha must be between 0 and 1"
assert torch.cuda.is_available() or (not misc.USE_GPU), "attempted to use gpu when unavailable"

# Define feature extractor
cnn = misc.to_device(Vgg16Pretrained())
phi = lambda x, y, z: cnn.forward(x, inds=y, concat=z)

# Load images
content_im_orig = misc.to_device(load_path_for_pytorch(content_path, target_size=sz)).unsqueeze(0)
style_im_orig = misc.to_device(load_path_for_pytorch(style_path, target_size=sz)).unsqueeze(0)

# Run Style Transfer
torch.cuda.synchronize()
start_time = time.time()
output = produce_stylization(content_im_orig, style_im_orig, phi,
                            max_iter=100,
                            lr=2e-3,
                            content_weight=content_weight,
                            max_scls=max_scls,
                            flip_aug=flip_aug,
                            content_loss=content_loss,
                            dont_colorize=dont_colorize)
torch.cuda.synchronize()
print('Done! total time: {}'.format(time.time() - start_time))

# Convert from pyTorch to numpy, clip to valid range
new_im_out = np.clip(output[0].permute(1, 2, 0).detach().cpu().numpy(), 0., 1.)

# Save stylized output
save_im = (new_im_out * 255).astype(np.uint8)
imwrite(output_path, save_im)

# Free gpu memory in case something else needs it later
if misc.USE_GPU:
    torch.cuda.empty_cache()
