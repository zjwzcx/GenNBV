# setup.py in legged_gym project

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


# Define the setup configuration
setup(
    name='legged_gym',
    version='1.0.0',
    author='Nikita Rudin',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='rudinn@ethz.ch',    # Credit to original author
    description='Isaac Gym environments',
    install_requires=[
        'isaacgym',
        'gym',
        'matplotlib',
        "tensorboard",
        "cloudpickle",
        "pandas",
        "yapf~=0.30.0",
        "wandb",
        "opencv-python>=3.0.0"
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
