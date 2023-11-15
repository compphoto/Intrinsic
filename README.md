# Intrinsic Image Decomposition via Ordinal Shading
Code for the paper: Intrinsic Image Decomposition via Ordinal Shading, [Chris Careaga](https://ccareaga.github.io/) and [Yağız Aksoy](https://yaksoy.github.io) , ACM Transactions on Graphics, 2023 
### [Project Page](https://yaksoy.github.io/intrinsic) | [Paper]() | [Video]() | [Supplementary]() | [Data]()

In this work, we achieve high-resolution intrinsic
decomposition by breaking the problem into two parts. First, we present a
dense ordinal shading formulation using a shift- and scale-invariant loss in
order to estimate ordinal shading cues without restricting the predictions to
obey the intrinsic model. We then combine low- and high-resolution ordinal
estimations using a second network to generate a shading estimate with both
global coherency and local details. We encourage the model to learn an ac-
curate decomposition by computing losses on the estimated shading as well
as the albedo implied by the intrinsic model. We develop a straightforward 
method for generating dense pseudo ground truth using our model’s pre-
dictions and multi-illumination data, enabling generalization to in-the-wild
imagery.

## Setup
Depending on how you would like to use the code in this repository there are two options to setup the code.
In either case, you should first create a fresh virtual environment (`python3 -m venv intrinsic_env`) and start it (`source intrinsic_env/bin/activate`)

#### Option 1
If you would like to download the repository to run and make changes you can simply clone the repo:
```
git clone https://github.com/compphoto/Intrinsic
cd Intrinsic
```
then pip install all the dependencies of the repo:
```
pip install -r requirements.txt 
```

#### Option 2
Alternatively, you can install this repository as a package using `setup.py`:
```
git clone https://github.com/compphoto/Intrinsic
cd Intrinsic
python setup.py
```
Or perform the same action without cloning the code using:
```
pip install https://github.com/compphoto/Intrinsic/archive/master.zip
```
This will allow you to import the repository as a python package, and use our pipeline as part of your codebase.

## Inference
To run our pipeline on your own images you can use the decompose script:
```

```
