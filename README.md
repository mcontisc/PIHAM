<h1 align="center">
PIHAM <br/>  
</h1>

<h3 align="center">
<i>Probabilistic Inference in Heterogeneous and Attributed Multilayer networks</i>
</h3>

<p align="center">
<a href="https://github.com/mcontisc/PIHAM/blob/main/LICENSE" target="_blank">
<img alt="License: MIT" src=https://img.shields.io/badge/License-MIT-green>
</a>

<a href="https://www.python.org/" target="_blank">
<img alt="Made with Python" src="https://img.shields.io/badge/made%20with-python-1f425f.svg">
</a>

<a href="https://arxiv.org/abs/2405.20918" target="_blank">
<img alt="ARXIV: 2301.11226" src="https://img.shields.io/badge/arXiv-2405.20918-red.svg">
</a>

</p>

This repository contains the implementation of the <i>PIHAM</i> model presented in 

&nbsp;&nbsp; 
[1] <i> Flexible inference in heterogeneous and attributed multilayer networks</i><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
Contisciani M., Hobbhahn M., Power E.A., Hennig P., and De Bacco C. (2024), <i> PNAS Nexus 4 (1), pgaf005</i><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[
        <a href="https://academic.oup.com/pnasnexus/article/4/1/pgaf005/7950749" target="_blank">PNAS Nexus</a>
]
[
        <a href="https://arxiv.org/abs/2405.20918" target="_blank">ArXiv</a>
]

If you make use of this code please cite our work in the form of the reference [1] above.

<h2> What's included </h2>

- `src`: Contains the Python implementation of the PIHAM algorithm, the code to generate synthetic data and additional utilities
- `data/input`: Contains a synthetic dataset generated using the PIHAM approach
- `data/output`: Contains some results

<h2> Requirements </h2>

In order to be able to run the code, you need to install the packages contained in `requirements.txt`. We suggest to create a conda environment with
`conda create --name PIHAM python=3.8 --no-default-packages`, activate it with `conda activate PIHAM`, and install all the dependencies by running (inside the `PIHAM` directory):

```bash
pip install -r requirements.txt
```

<h2> Perform inference </h2>

To perform the inference in a given heterogeneous and attributed multilayer network, run:

```bash
python main_inference.py
```

The script takes in input the name of the dataset, the path of the folder where it is stored, and the number of communities `K`.
It then executes the PIHAM algorithm from the file `src/model.py` using the configuration provided in the `src/setting_inference.yaml` file.

See the demo [jupyter notebook](http://localhost:8888/notebooks/Projects/SourceTree/_Public/PIHAM/analyse_results.ipynb) for an example on how to analyse the output results.

### Input format
The data should be stored in a `.pt` file, , which includes:

- `A`: An adjacency tensor of dimension L x N x N containing the interactions of every layer
- `X_categorical`: A design matrix with the categorical attribute 
- `X_poisson`: A design matrix with the Poisson attributes
- `X_gaussian`: A design matrix with the Gaussian attributes

Here, `L` is the number of layers and `N` is the number of nodes.

The code example in this directory is suitable to analyze a network with `L = 3` layers (one with binary interactions, 
the second with nonnegative discrete weights, and the third with real values) and three covariates (one categorical, 
one with nonnegative discrete values, and the last with real values). However, the model can be easily adapted to accommodate datasets with other data types.

### Output
The algorithm outputs a compressed file inside the `data/output` folder. To load the inferred results and display the out-going membership matrix, run:

```bash
import numpy as np 
theta = np.load("theta_<file_label>.npz")
print(theta["U"])
```

The variable `theta` includes the following parameters inferred by PIHAM: 

- `U`: The out-going membership matrix of dimension N x K
- `V`: The in-coming membership matrix of dimension N x K
- `W`: The affinity tensor of dimension L x K x K
- `Hcategorical`: The community-covariate matrix related to the categorical attribute of dimension K x Z_categorical
- `Hpoisson`: The community-covariate matrix related to the Poisson attributes of dimension K x P_poisson
- `Hgaussian`: The community-covariate matrix related to the Gaussian attribute of dimension K x P_gaussian
- `Cov`: The covariance matrix
- `Cov_diag`: The diagonal matrix of the variances

Here, `K` is the number of communities, `Z_categorical` is the number of categories for the categorical attribute, `P_poisson` is the number of Poisson attributes, and `P_gaussian` is the number of Gaussian attributes.

<h2> Run a cross-validation routine </h2>

If you are interested in assessing the prediction performance of PIHAM in a dataset for a given `K`, run:

```bash
python main_cv.py
```

The script takes in input the following parameters:

- `in_folder`: Path of the input folder
- `data_file`: Name of the dataset to analyse
- `K`: Number of communities 
- `NFold`: Number of folds for the cross-validation routine
- `cv_type`: Type of cross-validation routine
- `out_results`: Flag to save the prediction performance
- `--out_mask`: Flag to save the masks used during the cross-validation routine to hide entries of A and X
- `--out_inference`: Flag to save the inferred parameters during the cross-validation routine

For each fold, the script runs the PIHAM algorithm on the training set to learn its parameters, 
and evaluates its performance on the test set. This process is repeated `NFold` times, 
each time with a different fold as the test set. Various performance metrics are used depending on the type of information being evaluated. 
The results are saved in a `.csv` file in the `data/output/cv` folder.

<h2> Generate synthetic data </h2>
If you want to generate synthetic data using the PIHAM approach, run:

```bash
python main_generation.py
```

The script takes in input the number of independent samples to generate, a random seed, the number of communities `K`,
and the number of nodes `N`. The code example generates a heterogeneous and attributed network with `L = 3` layers (one with binary interactions, 
the second with nonnegative discrete weights, and the third with real values) and three covariates (one categorical, 
one with nonnegative discrete values, and the last with real values), using the default parameters specified in the file `src/synthetic.py`.
However, the script can be easily adapted to generate datasets with other data types and parameters.

