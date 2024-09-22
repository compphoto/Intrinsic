# Intrinsic Image Decomposition

This repository contains the code for the following papers: 

**Colorful Diffuse Intrinsic Image Decomposition in the Wild**, [Chris Careaga](https://ccareaga.github.io/) and [Yağız Aksoy](https://yaksoy.github.io), ACM Transactions on Graphics, 2024 \
(Paper and video coming soon!)

**Intrinsic Image Decomposition via Ordinal Shading**, [Chris Careaga](https://ccareaga.github.io/) and [Yağız Aksoy](https://yaksoy.github.io), ACM Transactions on Graphics, 2023 \
[Paper](https://yaksoy.github.io/papers/TOG23-Intrinsic.pdf) | [Video](https://www.youtube.com/watch?v=pWtJd3hqL3c) | [Supplementary](https://yaksoy.github.io/papers/TOG23-Intrinsic-Supp.pdf) | [Data](https://github.com/compphoto/MIDIntrinsics)
 
---


We propose a method for generating high-resolution intrinsic image decompositions, for in-the-wild images. Our method consists of multiple stages. We first estimate a grayscale shading layer using our ordinal shading pipeline. We then estimate low-resolution chromaticity information to account for color illumination effects while maintaining global consistency. Using this initial colorful decomposition, we estimate a high-resolution, sparse albedo layer. We show that our decomposition allows us to train a diffuse shading network using only a single rendered indoor dataset. 

![representative](./figures/representative.png)


Try out our pipeline on your own images! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/compphoto/Intrinsic/blob/main/intrinsic_inference.ipynb)

## Setup
Depending on how you would like to use the code in this repository there are two options to setup the code.
In either case, you should first create a fresh virtual environment (`python3 -m venv intrinsic_env`) and start it (`source intrinsic_env/bin/activate`)

You can install this repository as a package using `pip`:
```
git clone https://github.com/compphoto/Intrinsic
cd Intrinsic
pip install .
```
If you want to make changes to the code and have it reflected when you import the package use `pip install --editable`
Or perform the same action without cloning the code using:
```
pip install https://github.com/compphoto/Intrinsic/archive/main.zip
```
This will allow you to import the repository as a Python package, and use our pipeline as part of your codebase.

## Inference
To run our pipeline on your own images you can use the decompose script:
```python
from chrislib.general import view, tile_imgs, view_scale, uninvert
from chrislib.data_util import load_image

from intrinsic.pipeline import run_pipeline
from intrinsic.model_util import load_models

# load the models from the given paths
models = load_models('final_weights.pt')

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
shading = uninvert(inv_shd)

```
This will run our pipeline and output the linear albedo and shading. You can run this in your browser as well! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/compphoto/Intrinsic/blob/main/intrinsic_inference.ipynb)

## Citation

```
@ARTICLE{careagaIntrinsic,
  author={Chris Careaga and Ya\u{g}{\i}z Aksoy},
  title={Intrinsic Image Decomposition via Ordinal Shading},
  journal={ACM Trans. Graph.},
  year={2023},
}
```

## License

This implementation is provided for academic use only. Please cite our paper if you use this code or any of the models. 

The methodology presented in this work is safeguarded under intellectual property protection. For inquiries regarding licensing opportunities, kindly reach out to SFU Technology Licensing Office &#60;tlo_dir <i>ατ</i> sfu <i>δøτ</i> ca&#62; and Dr. Yağız Aksoy &#60;yagiz <i>ατ</i> sfu <i>δøτ</i> ca&#62;.
