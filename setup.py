from setuptools import find_packages, setup

setup(
    name="multiscale",  # change "multiscale" folder name to your project name
    packages=find_packages(".", exclude=["tests*"]),
)
