from chrislib.general import view, tile_imgs, invert
from chrislib.data_util import load_image

from intrinsic.pipeline import load_models, run_pipeline

# load the models from the given paths
models = load_models('v2')

# load an image (np float array in [0-1])
image = load_image('/path/to/input/image')

# run the model on the image (will use R0 resizing used in the ordinal shading paper by default)
results = run_pipeline(models, image, device='cuda')

img = results['image']
alb = view(results['hr_alb']) # gamma correct the estimated albedo
dif = 1 - invert(results['dif_shd']) # tonemap the diffuse shading
res = results['residual']

# save intrinsic components scaled to [0, 1] and gamma corrected
tile_imgs([img, alb, dif, res], save='output.png')


