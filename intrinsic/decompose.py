from chrislib.general import view, tile_imgs, view_scale, uninvert
from chrislib.data_util import load_image

from intrinsic.pipeline import run_pipeline
from intrinsic.model_util import load_models

# load the models from the given paths
models = load_models(
    ord_path='./final_weights/vivid_bird_318_300.pt',
    mrg_path='./final_weights/fluent_eon_138_200.pt'
)

# load an image (np float array in [0-1])
image = load_image('/path/to/input/image')

# run the model on the image using R_0 resizing
results = run_pipeline(
    models,
    image,
    resize_conf=0.0,
    maintain_size=True
)

albedo = results['albedo']
inv_shd = results['inv_shading']

# compute shading from inverse shading
our_shd = uninvert(inv_shd)

# save intrinsic components scaled to [0, 1] and gamma corrected
tile_imgs([image, view(albedo), view(our_shd)], save='output.png')
