from setuptools import find_packages, setup

import os

# get __version__
pkg_info = {}
here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, "prunhild", "_version.py")).read(), pkg_info)

readme = open("README.md").read()

requirements = {"install": ["torch", "torchvision"]}
install_requires = requirements["install"]

setup(
    # Metadata
    name="prunhild",
    version=pkg_info["__version__"],
    author="Peter Ruch",
    author_email="gfrogat@gmail.com",
    url="https://github.com/gfrogat/prunhild",
    license="MIT",
    description="Neural Network pruning libary for PyTorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    # Package info
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
)
