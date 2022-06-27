#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.readlines()

version = "0.0.0"

setup(
    name="zeroshot_classification",
    version=version,
    description="A package for my thesis on Evaluating Zero-shot Classification Methods in Turkish.",
    python_requires=">=3.8",
    author="emrecncelik",
    author_email="emrecncelik@gmail.com",
    license="MIT License",
    url="https://github.com/emrecncelik/zeroshot-turkish.git",
    download_url="https://github.com/emrecncelik/zeroshot-turkish.git",
    install_requires=[requirements],
    long_description=readme,
    include_package_data=True,
    packages=find_packages(
        include=[
            "zeroshot_classification",
            "zeroshot_classification.*",
        ]
    ),
    zip_safe=False,
)
