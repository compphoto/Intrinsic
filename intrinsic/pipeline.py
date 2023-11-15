import torch
import numpy as np
from skimage.transform import resize

from chrislib.resolution_util import optimal_resize
from chrislib.general import round_32, uninvert

from intrinsic.ordinal_util import base_resize, equalize_predictions


def run_pipeline(
        models,
        img_arr,
        output_ordinal=False,
        resize_conf=0.0,
        base_size=384,
        maintain_size=False,
        linear=False,
        device='cuda',
        lstsq_p=0.0,
        inputs='all'):
    """Runs the complete pipeline for shading and albedo prediction

    params:
        models (dict): models dictionary returned by model_util.load_models()
        img_arr (np.array): RGB input image as numpy array between 0-1
        output_ordinal (bool) optional: whether or not to output intermediate ordinal estimations
            (default False)
        resize_conf (float) optional: confidence to use for resizing (between 0-1) if None maintain
            original size (default None)
        base_size (int) optional: size of the base resolution estimation (default 384)
        maintain_size (bool) optional: whether or not the results match the input image size
            (default False)
        linear (bool) optional: whether or not the input image is already linear (default False)
        device (str) optional: string representing device to use for pipeline (default "cuda")
        lstsq_p (float) optional: subsampling factor for computing least-squares fit 
            when matching the scale of base and full estimations (default 0.0)
        inputs (str) optional: network inputs ("full", "base", "rgb", "all") the rgb image is
            always included (default "all")

    returns:
        results (dict): a result dictionary with albedo, shading and potentiall ordinal estimations
    """
    results = {}

    orig_h, orig_w, _ = img_arr.shape
    
    # if no confidence value set, just round original size to 32 for model input
    if resize_conf is None:
        img_arr = resize(img_arr, (round_32(orig_h), round_32(orig_w)), anti_aliasing=True)

    # if a the confidence is an int, just rescale image so that the large side
    # of the image matches the specified integer value
    elif isinstance(resize_conf, int):
        scale = resize_conf / max(orig_h, orig_w)
        img_arr = resize(
            img_arr,
            (round_32(orig_h * scale), round_32(orig_w * scale)),
            anti_aliasing=True)
    
    # if the confidence is a float use the optimal resize code from Miangoleh et al.
    elif isinstance(resize_conf, float):
        img_arr = optimal_resize(img_arr, conf=resize_conf)

    fh, fw, _ = img_arr.shape
    
    # if the image is in sRGB we do simple linearization using gamma=2.2
    if not linear:
        lin_img = img_arr ** 2.2
    else:
        lin_img = img_arr

    with torch.no_grad():
        # ordinal shading estimation --------------------------
        
        # resize image for base and full estimations and send through ordinal net
        base_input = base_resize(lin_img, base_size)
        full_input = lin_img

        base_input = torch.from_numpy(base_input).permute(2, 0, 1).to(device).float()
        full_input = torch.from_numpy(full_input).permute(2, 0, 1).to(device).float()

        base_out = models['ordinal_model'](base_input.unsqueeze(0)).squeeze(0)
        full_out = models['ordinal_model'](full_input.unsqueeze(0)).squeeze(0)
        
        # the ordinal estimations come out of the model with a channel dim
        base_out = base_out.permute(1, 2, 0).cpu().numpy()
        full_out = full_out.permute(1, 2, 0).cpu().numpy()

        base_out = resize(base_out, (fh, fw))

        # if we are using all inputs, we scale the input estimations using the base estimate
        if inputs == 'all':
            ord_base, ord_full = equalize_predictions(lin_img, base_out, full_out, p=lstsq_p)
        else:
            ord_base, ord_full = base_out, full_out
        # ------------------------------------------------------

        # ordinal shading to real shading ----------------------
        inp = torch.from_numpy(lin_img).permute(2, 0, 1).to(device)
        bse = torch.from_numpy(ord_base).permute(2, 0, 1).to(device)
        fll = torch.from_numpy(ord_full).permute(2, 0, 1).to(device)
        
        # combine the base and full ordinal estimations w/ the input image
        if inputs == 'full':
            combined = torch.cat((inp, fll), 0).unsqueeze(0)
        elif inputs == 'base':
            combined = torch.cat((inp, bse), 0).unsqueeze(0)
        elif inputs == 'rgb':
            combined = inp.unsqueeze(0)
        else:
            combined = torch.cat((inp, bse, fll), 0).unsqueeze(0)

        inv_shd = models['real_model'](combined)
        
        # the shading comes out in the inverse space so undo it 
        shd = uninvert(inv_shd)
        alb = inp / shd
        # ------------------------------------------------------
    
    # put all the outputs into a dictionary to return
    inv_shd = inv_shd.squeeze(0).detach().cpu().numpy()
    alb = alb.permute(1, 2, 0).detach().cpu().numpy()

    if maintain_size:
        if output_ordinal:
            ord_base = resize(base_out, (orig_h, orig_w), anti_aliasing=True)
            ord_full = resize(full_out, (orig_h, orig_w), anti_aliasing=True)

        inv_shd = resize(inv_shd, (orig_h, orig_w), anti_aliasing=True)
        alb = resize(alb, (orig_h, orig_w), anti_aliasing=True)

    if output_ordinal:
        results['ord_full'] = ord_full
        results['ord_base'] = ord_base

    results['inv_shading'] = inv_shd
    results['albedo'] = alb
    results['image'] = img_arr

    return results
