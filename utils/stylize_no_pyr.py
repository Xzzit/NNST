# Core Dependencies
import random

# External Dependency Imports
import torch
from torch.autograd import Variable

# Internal Project Imports
from utils.distance import pairwise_distances_gram, pairwise_distances_cos_center
from utils.featureExtract import extract_feats, get_feat_norms
from utils import misc
from utils.misc import to_device, flatten_grid, scl_spatial
from utils.colorization import color_match


def produce_stylization(content_img, style_img, phi,
                        max_iter=350, lr=1e-3,
                        content_weight=1., max_scls=4,
                        flip_aug=False, content_loss=False,
                        zero_init=False, dont_colorize=False, top_k=1):
    """ Produce stylization of 'content_img' in the style of 'style_img'
        Inputs:
            content_img -- 1x3xHxW pytorch tensor containing rbg content image
            style_img -- [1x3xH'xW', ..,] pytorch tensor containing rgb style images
            phi -- lambda function to extract features using VGG16Pretrained
            max_iter -- number of updates to image pyramid per scale
            lr -- learning rate of optimizer updating pyramid coefficients
            content_weight -- controls stylization level, between 0 and 1
            max_scl -- number of scales to stylize (performed coarse to fine)
            flip_aug -- extract features from rotations of style image too?
            content_loss -- use self-sim content loss? (compares down sampled
                            version of output and content image)
            zero_init -- if true initialize w/ grey image, o.w. initialize w/
                         down sampled content image

        Output:
            stylized image
    """

    # Initialize output image
    if zero_init:
        # Initialize with flat grey image (works, but less vivid)
        output_img = torch.zeros_like(content_img) + 0.5

    else:
        output_img = content_img.clone()

    # Stylize using hypercolumn matching from coarse to fine scale
    li = 0
    for scl in range(max_scls)[::-1]:  # From end to start

        # Get content image and style image at current resolution
        if misc.USE_GPU:
            torch.cuda.empty_cache()

        style_im_tmp = []
        for s in style_img:
            s = scl_spatial(s, s.size(2) // 2 ** (scl), s.size(3) // 2 ** (scl))
            style_im_tmp.append(s)
            
        content_im_tmp = scl_spatial(content_img,
                                     content_img.size(2) // 2 ** (scl),
                                     content_img.size(3) // 2 ** (scl))
        output_img = scl_spatial(output_img,
                                 content_img.size(2) // 2 ** (scl),
                                 content_img.size(3) // 2 ** (scl))
        li += 1
        print(f'-{li, max(output_img.size(2), output_img.size(3))}-')

        # Construct stylized activations
        with torch.no_grad():

            # Control tradeoff between searching for features that match current iterate,
            # and features that match content image (at coarsest scale, only use content image)
            alpha = content_weight
            if li == 1:
                alpha = 0.

            # Search for features using high frequencies from content (but do not initialize actual output with them)
            # Extract style features from rotated copies of style image
            feats_s = []
            for s in style_im_tmp:
                s = extract_feats(s, phi, flip_aug=flip_aug).cpu()
                feats_s.append(s)

            # Extract features from convex combination of content image and current iterate:
            c_tmp = (output_img * alpha) + (content_im_tmp * (1. - alpha))
            feats_c = extract_feats(c_tmp, phi).cpu()

            # Replace content features with style features
            target_feats = replace_features(feats_c, feats_s, top_k=top_k)

        # Synthesize output at current resolution using hypercolumn matching
        output_img = optimize_output_im(output_img, content_img, style_img,
                                           target_feats, lr, max_iter, scl, phi,
                                           content_loss=content_loss)

    # Perform final pass using feature splitting (pass in flip_aug argument
    # because style features are extracted internally in this regime)
    output_img = optimize_output_im(output_img, content_img, style_img,
                                       target_feats, lr, max_iter, scl, phi,
                                       final_pass=True, content_loss=content_loss,
                                       flip_aug=flip_aug)

    if dont_colorize:
        return output_img
    else:
        return color_match(content_img, style_img, output_img)


def replace_features(src, ref, top_k=1):
    """ Replace each feature vector in 'src' with the nearest (under centered
    cosine distance) feature vector in 'ref'
    Inputs:
        src -- [1, C, H, W] tensor of content features
        ref -- [1, C, H', W'] tensor of style features
    Outputs:
        rplc -- [1, C, H, W] tensor of features, where rplc[0,:,i,j] is the nearest
                neighbor feature vector of src[0,:,i',j'] in ref
    """
    # Move style features, content features to gpu
    src_flat_all = flatten_grid(src)  # Shape: [(H' * W'), C: e.g. 2688]
    ref_flat = to_device(flatten_grid(ref))  # Shape: [(H * W), C: e.g. 2688]

    # How many rows of the distance matrix to compute at once, can be
    # reduced if less memory is available, but this slows method down
    # Stride becomes smaller when style image size getting bigger
    stride = 128 ** 2 // max(1, (src.size(2) * src.size(3)) // (128 ** 2))
    bi = 0

    # Loop until all content features are replaced by style feature / all rows of distance matrix are computed
    out = []
    while bi < src_flat_all.size(0):
        ei = min(bi + stride, src_flat_all.size(0))

        # Get chunk of content features, compute corresponding portion
        # distance matrix, and store nearest style feature to each content feature
        src_flat = to_device(src_flat_all[bi:ei, :])
        d_mat = pairwise_distances_cos_center(ref_flat, src_flat)

        # Get top k nearest neighbor indices
        _, nn_inds = torch.topk(d_mat, top_k, dim=0, largest=False, sorted=False)

        del d_mat  # distance matrix uses lots of memory, free asap

        # Get style feature closest to each content feature and save in 'out'
        for i in nn_inds:
            ref_sel = torch.index_select(ref_flat, 0, i).transpose(1, 0).contiguous()
        ref_sel /= nn_inds.size(0)
        out.append(ref_sel)  # .view(1, ref.size(1), src.size(2), ei - bi))

        bi = ei

    out = torch.cat(out, 1)
    out = out.view(1, src.size(1), src.size(2), src.size(3))

    return out


def optimize_output_im(output_img, content_img, style_img, target_feats,
                       lr, max_iter, scl, phi, final_pass=False,
                       content_loss=False, flip_aug=True):
    """ Optimize stylized image at a given resolution
        Inputs:
            output_img -- output image
            content_img -- content image
            style_img -- style image
            target_feats -- precomputed target features of stylized output
            lr -- learning rate for optimization
            max_iter -- maximum number of optimization iterations
            scl -- integer controls which resolution to optimize
            phi -- lambda function to compute features using pretrained VGG16
            final_pass -- if true, ignore 'target_feats' and recompute target
                          features before every step of gradient descent (and
                          compute feature matches separately for each layer
                          instead of using hypercolumns)
            content_loss -- if true, also minimize content loss that maintains
                            self-similarity in color space between 32pixel
                            downsampled output image and content image
            flip_aug -- if true, extract style features from rotations of style
                        image. This increases content preservation by making
                        more options available when matching style features
                        to content features
        Outputs:
            output_pyr -- stylized output image at target resolution
    """
    
    # Initialize optimizer variables and optimizer
    opt_vars = output_img.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_vars], lr=lr)

    # Original features uses all layers, but dropping conv5 block  speeds up method without hurting quality
    feature_list_final = [22, 20, 18, 15, 13, 11, 8, 6, 3, 1]

    # Precompute features that remain constant
    if not final_pass:
        # Precompute normalized features targets during hypercolumn-matching regime for cosine distance
        target_feats_n = target_feats / get_feat_norms(target_feats)

    else:
        # For feature-splitting regime extract style features for each conv
        # layer without downsampling (including from rotations if applicable)
        s_feat = []
        for s in style_img:
            s_feat += [phi(s, feature_list_final, False)]
        s_feat = [list(x) for x in zip(*s_feat)]

        if flip_aug:
            aug_list = [torch.flip(style_img, [2]).transpose(2, 3),
                        torch.flip(style_img, [2, 3]),
                        torch.flip(style_img, [3]).transpose(2, 3)]

            for ia, im_aug in enumerate(aug_list):
                s_feat_tmp = phi(im_aug, feature_list_final, False)

                if ia != 1:
                    s_feat_tmp = [s_feat_tmp[iii].transpose(2, 3)
                                  for iii in range(len(s_feat_tmp))]

                s_feat = [torch.cat([s_feat[iii], s_feat_tmp[iii]], 2)
                          for iii in range(len(s_feat_tmp))]

    # Precompute content self-similarity matrix if needed for 'content_loss'
    if content_loss:
        c_scl = max(content_img.size(2), content_img.size(3))
        c_fac = c_scl // 32
        h = int(content_img.size(2) / c_fac)
        w = int(content_img.size(3) / c_fac)

        content_im_tmp = scl_spatial(content_img,
                                     content_img.size(2) // 2 ** (scl + 1),
                                     content_img.size(3) // 2 ** (scl + 1))  # Get original content image

        c_low_flat = flatten_grid(scl_spatial(content_im_tmp, h, w))
        self_sim_target = pairwise_distances_gram(c_low_flat, c_low_flat).clone().detach()

    # Optimize pixels to find image that produces stylized activations
    for i in range(max_iter):

        # Zero out gradient and loss before current iteration
        optimizer.zero_grad()
        ell = 0.

        # Compare current features with stylized activations
        if not final_pass:  # hypercolumn matching / 'hm' regime

            # Extract features from current output, normalize for cos distance
            out_feats = extract_feats(opt_vars, phi)
            out_feats_n = out_feats / get_feat_norms(out_feats)

            # Update overall loss w/ cosine loss w.r.t target features
            ell = ell + (1. - (target_feats_n * out_feats_n).sum(1)).mean()


        else:  # feature splitting / 'fs' regime
            # Extract features from current output (keep each layer seperate and don't downsample)
            out_feats = phi(opt_vars, feature_list_final, False)

            # Compute matches for each layer. For efficiency don't explicitly
            # gather matches, only access through distance matrix.
            ell_fs = 0.
            for h_i in range(len(s_feat)):
                # Get features from a particular layer
                s_tmp = s_feat[h_i]
                out_temp = out_feats[h_i]
                chans = out_temp.size(1)

                # Sparsely sample feature tensors if too big, otherwise just reshape
                if max(out_temp.size(2), out_temp.size(3)) > 64:
                    stride = max(out_temp.size(2), out_temp.size(3)) // 64
                    offset_a = random.randint(0, stride - 1)
                    offset_b = random.randint(0, stride - 1)

                    # combine all of the style features into one tensor
                    s_samp = []
                    for s in s_tmp:
                        s_samp += s[:, :, offset_a::stride, offset_b::stride]
                    s_samp = [s.contiguous().view(1, chans, -1) for s in s_samp]
                    s_samp = torch.cat(s_samp, 2)

                    out_samp = out_temp[:, :, offset_a::stride, offset_b::stride]
                    out_samp = out_samp.contiguous().view(1, chans, -1)

                else:
                    s_samp = flatten_grid(s_tmp).transpose(1, 0).unsqueeze(0)
                    out_samp = flatten_grid(out_temp).transpose(1, 0).unsqueeze(0)

                # Compute distance matrix and find minimum along each row to
                # implicitly get matches (and minimize distance between them)
                d_mat = pairwise_distances_cos_center(s_samp[0].transpose(1, 0),
                                                      out_samp[0].transpose(1, 0))
                d_min, _ = torch.min(d_mat, 0)

                # Aggregate loss over layers
                ell_fs += d_min.mean()

            # Update overall loss
            ell = ell + ell_fs

        # Optional self similarity content loss between downsampled output
        # and content image. Always turn off at end for best results.
        if content_loss and not (final_pass and i > 100):
            o_flat = flatten_grid(scl_spatial(opt_vars, h, w))
            self_sim_out = pairwise_distances_gram(o_flat, o_flat)

            ell = ell + torch.mean(torch.abs((self_sim_out - self_sim_target)))

        # Update output's pyramid coefficients
        ell.backward()
        optimizer.step()

    return opt_vars
