# Core Imports
import os
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
content_dir = 'D:/Project/CPro/instant-ngp/data/nerf/real_world/mountain_1/baseline'
style_img = 'D:/Project/Pypro/data/art/inkmoun.jpg'
output_path = 'D:/Project/CPro/instant-ngp/data/nerf/real_world/mountain_1/images'
max_scls = 4
sz = 1080
flip_aug = False
content_loss = False
misc.USE_GPU = True
content_weight = 0.5

# Error checking for arguments
# error checking for paths deferred to imageio
assert (0.0 <= content_weight) and (content_weight <= 1.0), "alpha must be between 0 and 1"
assert torch.cuda.is_available() or (not misc.USE_GPU), "attempted to use gpu when unavailable"

# Define feature extractor
cnn = misc.to_device(Vgg16Pretrained())
phi = lambda x, y, z: cnn.forward(x, inds=y, concat=z)


dirlist = os.listdir(content_dir)
for name in dirlist:
    content_img = os.path.join(content_dir, name)

    # Load images
    content_im_orig = misc.to_device(load_path_for_pytorch(content_img, target_size=sz)).unsqueeze(0)
    style_im_orig = misc.to_device(load_path_for_pytorch(style_img, target_size=sz)).unsqueeze(0)

    # Run Style Transfer
    torch.cuda.synchronize()
    start_time = time.time()
    output = produce_stylization(content_im_orig, style_im_orig, phi,
                                max_iter=200,
                                lr=2e-3,
                                content_weight=content_weight,
                                max_scls=max_scls,
                                flip_aug=flip_aug,
                                content_loss=content_loss,
                                dont_colorize=False)
    torch.cuda.synchronize()
    print('Done! total time: {}'.format(time.time() - start_time))

    # Convert from pyTorch to numpy, clip to valid range
    new_im_out = np.clip(output[0].permute(1, 2, 0).detach().cpu().numpy(), 0., 1.)

    # Save stylized output
    save_im = (new_im_out * 255).astype(np.uint8)
    imwrite(os.path.join(output_path, name), save_im)

    # Free gpu memory in case something else needs it later
    if misc.USE_GPU:
        torch.cuda.empty_cache()
