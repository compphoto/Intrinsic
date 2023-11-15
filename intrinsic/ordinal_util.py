import torch
import numpy as np
from skimage.transform import resize

from chrislib.general import round_32, minmax, get_brightness

def base_resize(img, base_size=384):
    """TODO DESCRIPTION

    params:
        img (TODO): TODO
        base_size (int) optional: TODO (default 384)

    returns:
        net_input (TODO): TODO
    """
    h, w, _ = img.shape

    max_dim = max(h, w)
    scale = base_size / max_dim

    new_h, new_w = scale * h, scale * w
    new_h, new_w = round_32(new_h), round_32(new_w)

    net_input = resize(img, (new_h, new_w, 3), anti_aliasing=True)
    return net_input


# def full_resize(img):
#     """TODO DESCRIPTION
# 
#     params:
#         img (TODO): TODO
# 
#     returns:
#         net_input (TODO): TODO
#     """
#     h, w, _ = img.shape
#     new_h, new_w = round_32(h), round_32(w)
# 
#     net_input = resize(img, (new_h, new_w, 3), anti_aliasing=True)
#     return net_input


def equalize_predictions(img, base, full, p=0.5):
    """TODO DESCRIPTION

    params:
        img (TODO): TODO
        base (TODO): TODO
        full (TODO): TODO
        p (int) optional: TODO (default 0.5)

    returns:
        base (TODO): TODO
        new_full (TODO): TODO
    """
    h, w, _ = img.shape

    full_shd = (1. / full.clip(1e-5)) - 1.
    base_shd = (1. / base.clip(1e-5)) - 1.
    
    full_alb = get_brightness(img) / full_shd.clip(1e-5)
    base_alb = get_brightness(img) / base_shd.clip(1e-5)

    rand_msk = (np.random.randn(h, w) > p).astype(np.uint8)

    flat_full_alb = full_alb[rand_msk == 1]
    flat_base_alb = base_alb[rand_msk == 1]

    scale, _, _, _ = np.linalg.lstsq(flat_full_alb.reshape(-1, 1), flat_base_alb, rcond=None)

    new_full_alb = scale * full_alb
    new_full_shd = get_brightness(img) / new_full_alb.clip(1e-5)
    new_full = 1.0 / (1.0 + new_full_shd)

    return base, new_full


def ordinal_forward(model, img, normalize=False, dev='cuda'):
    """TODO DESCRIPTION

    params:
        model (TODO): TODO
        img (TODO): TODO
        normalize (bool) optional: TODO (default False)
        dev (str) optional: the device to run the model on (default "cuda")

    returns:
        (TODO): TODO
    """
    fh, fw, _ = img.shape

    base_input = base_resize(img)
    full_input = full_resize(img)

    base_input = torch.from_numpy(base_input).permute(2, 0, 1).to(dev).float()
    full_input = torch.from_numpy(full_input).permute(2, 0, 1).to(dev).float()

    with torch.no_grad():
        base_out = model(base_input.unsqueeze(0)).squeeze(0)
        full_out = model(full_input.unsqueeze(0)).squeeze(0)

    if normalize:
        base_out = minmax(base_out)
        full_out = minmax(full_out)

    base_out = base_out.unsqueeze(-1).cpu().numpy()
    full_out = full_out.cpu().unsqueeze(-1).numpy()

    base_est = resize(base_out, (fh, fw))
    full_est = resize(full_out, (fh, fw))

    return equalize_predictions(img, base_est, full_est)
