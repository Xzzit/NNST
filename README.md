[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# NNST
This is an updated repository for the NNST-Opt algorithm, which is described in the paper [Neural Neighbor Style Transfer](https://ttic.uchicago.edu/~nickkolkin/Paper/NNST_Preprint.pdf). The original code can be found at [nkolkin13](https://github.com/nkolkin13/NeuralNeighborStyleTransfer). Try Replicate web demo here [![Replicate](https://replicate.com/nkolkin13/neuralneighborstyletransfer/badge)](https://replicate.com/nkolkin13/neuralneighborstyletransfer)

## What's New
* Update to the Pytorch 1.13 version.
* Deprecate the usage of the image pyramid architecture, as it significantly improves code readability.
* Introduce a new optional flag, `--top x`, which allows the selection of the xth nearest neighbor for replacing the feature maps. This enhancement improves the stability and visual consistency of the generated results.
* Set the default value of the flip argument to false, which improves the inference speed (from 21s to 16s, in the case of RTX 3090), while resulting in minimal quality loss.
* I have added a `show_output.ipynb` file to the repository. This file contains examples and explanations of the usage of various functions that appear in the original code. It is designed to help you understand the code more quickly and efficiently. You can refer to this file to enhance your comprehension of the codebase and improve your overall speed in understanding the code.

## Dependencies
Tested With:
* Python 3.10
* Pytorch 1.13
* Imageio 2.31

## Example Output -Image
Example output produced using included files with the command:
```
python styleTransfer.py --content_path inputs/content/C1.png --style_path inputs/style/S4.jpg --output_path ./output.jpg
```
![Example Output](https://github.com/nkolkin13/NeuralNeighborStyleTransfer/blob/main/example2.png?raw=true)

To produce an output without color correction use the command
```
python styleTransfer.py --content_path inputs/content/C1.png --style_path inputs/style/S4.jpg --output_path ./output.jpg --dont_colorize
```
![Example Output w/o Colorization](https://github.com/nkolkin13/NeuralNeighborStyleTransfer/blob/main/example.png?raw=true)

## Examples Output -Video
https://home.ttic.edu/~nickkolkin/nnst_video_supp.mp4

## Hardware Requirements
Primarily tested with Nvidia GPU using CUDA, CPU mode implemented but not tested extensively (and is very slow).  
Generating 512x512 outputs requires ~6GB of memory, and generating 1024x1024 outputs requires ~12GB of memory.    

## Usage
Default Settings (512x512 Output):    
```
python styleTransfer.py --content_path PATH_TO_CONTENT_IMAGE --style_path PATH_TO_STYLE_IMAGE --output_path PATH_TO_OUTPUT
```
(Optional Flag) Producing 1024x1024 Output. 

N.B: to get the most out of this setting use styles that are at least 1024 pixels on the long side, the included styles are too small (512 pixels on the long side):    
```
python styleTransfer.py --content_path PATH_TO_CONTENT_IMAGE --style_path PATH_TO_STYLE_IMAGE --output_path PATH_TO_OUTPUT --high_res
```

(Optional Flag) Set Alpha, must be between 0.0 and 1.0. Alpha=1.0 corresponds to maximum content preservation, and Alpha=0.0 is maximum stylization  (Default is 0.75):    
```
python styleTransfer.py --content_path PATH_TO_CONTENT_IMAGE --style_path PATH_TO_STYLE_IMAGE --output_path PATH_TO_OUTPUT --alpha ALPHA_VALUE
```

(Optional Flag) Augment style image with rotations. Slows down the algorithm and increases memory requirement. Generally improves content preservation but hurts stylization slightly:  
```
python styleTransfer.py --content_path PATH_TO_CONTENT_IMAGE --style_path PATH_TO_STYLE_IMAGE --output_path PATH_TO_OUTPUT --flip
```

(Optional Flag) Cpu Mode, takes tens of minutes even for a 512x512 output:  
```
python styleTransfer.py --content_path PATH_TO_CONTENT_IMAGE --style_path PATH_TO_STYLE_IMAGE --output_path PATH_TO_OUTPUT --cpu
```

(Optional Flag) Use experimental content loss. The most common failure mode of our method is that colors will shift within an object creating highly visible artifacts, if that happens this flag can usually fix it, but currently, it has some drawbacks which is why it isn't enabled by default (see below for details). One advantage of using this flag though is that Alpha can typically be set all the way to 0.0 and the content will remain recognizable:  
```
python styleTransfer.py --content_path PATH_TO_CONTENT_IMAGE --style_path PATH_TO_STYLE_IMAGE --output_path PATH_TO_OUTPUT --content_loss
```

Optional Flags can be combined.

## Experimental Content Loss
Because by default our method doesn't use a content loss it sometimes destroys important content details, especially if alpha is below 0.5. I'm currently working on a minimally invasive content loss based on the self-similarity matrix of the downsampled content image. So far it seems to reliably ensure the content is preserved, but it has two main flaws. 

The first is that it causes periodic artifacts (isolated bright or dark pixels at regular intervals), which is caused by using bilinear downsampling. I'm working on a modified downsampler that randomizes the blur kernel which should fix this, and I'll update this repo when it's ready.

The second is that for styles with limited palettes (typically drawing-based styles), the content loss will cause unfaithful colors that interpolate between the limited palette, I'm still thinking of a way to address this.

## Included Example Inputs
The most important thing about choosing input style and content images is ensuring that they are at least as high resolution as the output image you want (this is most important for the style image, but very helpful for the content image as well). 

Generally, I've found that style images that work well for one image, tend to work well for many images, I've included some examples of such images in ./inputs/style/. If a style image consists of large visual elements (for example large shapes in a cubist painting), our method is less likely to capture it. Sometimes setting Alpha to be near 1.0 will work, but this isn't guaranteed.

The content images that work the best are ones with a single large object. The smaller or lower contrast an aspect of the content image is, the more likely it will be lost in the final output. I've included some examples of content images that work well in ./inputs/content/
