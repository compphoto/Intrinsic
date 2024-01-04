from chrislib.general import view, tile_imgs, view_scale, uninvert
from chrislib.data_util import load_image

from intrinsic.pipeline import run_pipeline
from intrinsic.model_util import load_models

# load the models from the given paths
models = load_models('paper_weights')

# load an image (np float array in [0-1])
image = load_image('/path/to/input/image')

# run the model on the image using R_0 resizing 
# the maintain_size will output components that match the original image dims
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
