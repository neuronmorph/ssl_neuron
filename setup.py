from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
#     "allensdk",
#     "numpy",
#     "tqdm",
#     "seaborn",
#     "matplotlib",
#     "networkx==2.5",
#     "scipy==1.6.0",
#     "scikit-learn==0.24.1",
#     "torch==1.13.1",
#     "torchvision"
]


setup(name='ssl_neuron', 
        version='1.0',
        description="SSL-Neuron contains the code to the paper 'Self-supervised Representation Learning of Neuronal Morphologies'",
        author="Marissa Weis",
        author_email="marissa.weis@bethgelab.org",
        url="https://eckerlab.org/code/weis2021b/",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3"
                 ],
        packages=find_packages(),
        install_requires=install_requires,
        )