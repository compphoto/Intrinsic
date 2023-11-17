import setuptools
setuptools.setup(
    name="intrinsic",
    version="0.0.1",
    author="Chris Careaga",
    author_email="chris_careaga@sfu.ca",
    description='a package containing to the code for the paper "Intrinsic Image Decomposition via Ordinal Shading"',
    url="",
    packages=setuptools.find_packages(),
    license="",
    python_requires=">3.6",
    install_requires=[
    'altered_midas @ git+https://github.com/CCareaga/MiDaS@master',
    'chrislib @ git+https://github.com/CCareaga/chrislib@main'
    ]
)
