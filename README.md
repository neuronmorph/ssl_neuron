# Positional Embeddings for GraphDINO

This repository contains different positional embeddings for GraphDINO. Positional embeddings are organized by tags:
1. pstepRWPE+euclPE  - p-step random walk matrix and eigenvectors of euclidean distance matrix
2. lapPE_only  - eigenvectros of Laplacian matrix
3. pstepRWPE_only - p-step random walk matrix
4. eucl_disPE_only - eigenvectors of euclidean distance matrix
5. cond_distPE_only - eigenvectors of conductance distance matrix
6. allPE - all above positional embeddings combined
7. with_cond_dist_in_GT - includes the previous version of positional embeddings, which were summed to node features instead of concatenating. Disregard this tag.

## Run inference
For model inference use Checkpoints_visualized.ipynb. 
1. Load the corresponding code using the tags above. 
2. Select the correct config file and change the paths to checkpoint files
3. Uncomment the cell with the positional embedding(s)
4. Change the CUDA device number in train.py line 10 to "cuda:0"; and in graphdino.py line 418 to "cuda:0"
5. Run the notebook. It will give accuracy score, confusion matrix, and t-SNE clustering

## Model weights
Model weights are located in ckpts folder with the corresponding names 

## Self-supervised Representation Learning of Neuronal Morphologies

This repository contains code to the paper [Self-supervised Representation Learning of Neuronal Morphologies](https://arxiv.org/abs/2112.12482) by M.A. Weis, L. Pede, T. Lüddecke and A.S. Ecker (2021).

## Installation

```
python3 setup.py install
```

## Data

Extract data using the [Allen Software Development Kit](http://alleninstitute.github.io/AllenSDK/cell_types.html). See [demo notebook](http://alleninstitute.github.io/AllenSDK/_static/examples/nb/cell_types.html#Cell-Morphology-Reconstructions) on how to use the Allen Cell Types Database.

See [extract_allen_data.ipynb](https://github.com/marissaweis/ssl_neuron/blob/main/ssl_neuron/data/extract_allen_data.ipynb) for data preprocessing.


## Training
Start training GraphDINO from scratch on ABA dataset:
```
python3 ssl_neuron/main.py --config=ssl_neuron/configs/config.json
```

## Demos
For examples on how to load the data, train the model and perform inference with a pretrained model, see Jupyter notebooks in the [demos folder](https://github.com/marissaweis/ssl_neuron/tree/main/ssl_neuron/demos).


## Citation

If you use this repository in your research, please cite:
```
@article{Weis2021,
      title={Self-supervised Representation Learning of Neuronal Morphologies}, 
      author={Marissa A. Weis and Laura Pede and Timo Lüddecke and Alexander S. Ecker},
      year={2021},
      journal={arXiv}
}
```