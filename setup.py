from setuptools import setup, find_packages

setup(
    name="sir-gan",
    version="0.1.0",
    description="Synthetic IR Image Refinement using Adversarial Learning",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#")
    ],
    python_requires=">=3.8",
)
