from setuptools import setup
from setuptools import find_packages


VERSION = '0.1.6'

setup(
    name='sadtalker_z',  # package name
    version=VERSION,  # package version
    description='sadtalker',  # package description
    packages=find_packages("."),
    zip_safe=False,
    install_requires=[
        "numpy==1.23.4",
        "face_alignment>=1.3.5",
        "imageio>=2.19.3",
        "imageio-ffmpeg>=0.4.7",
        "librosa>=0.9.2",
        "numba",
        "resampy>=0.3.1",
        "pydub>=0.25.1 ",
        "scipy>=1.10.1",
        "kornia>=0.6.8",
        "tqdm",
        "yacs>=0.1.8",
        "pyyaml",
        "joblib>=1.1.0",
        "scikit-image>=0.19.3",
        "basicsr>=1.4.2",
        "facexlib>=0.3.0",
        "gradio",
        "gfpgan",
        "av",
        "safetensors",
        "pyaiy",
        "opencv-python"
    ]
)