# ValidMLInference

ValidMLInference is a Python package for estimating linear models which use synthetically generated regressors. The bias-correction methods are described in [Battaglia, Christensen, Hansen & Sacher (2024)](https://arxiv.org/abs/2402.15585). 

## Requirements and installation

`ValidMLInference` runs on Python 3.8 and requires a couple of standard numerical packages: numpy, scipy, jax, jaxopt, and numdifftools. The package can be installed `ValidMLInference` by typing ``` > pip install ValidMLInference ```  into the terminal. 

## Using ValidMLInference

To get started with using the package, we recommend looking at the following examples and resources: 
1. [Remote Work](https://github.com/KonradKurczynski/ValidMLInference/blob/main/remote_work.ipynb): this notebook contains an example of estimating the association between working from home and salaries in job postings using real-world data, reproducing results from Table 1 of [Battaglia, Christensen, Hansen & Sacher (2024)](https://arxiv.org/abs/2402.15585)
2. [Synthetic Example](https://github.com/KonradKurczynski/ValidMLInference/blob/main/synthetic_example.ipynb): this notebook contains a synthetic example comparing the performance of the different bias-correction methods
3. [Functionality](https://github.com/KonradKurczynski/ValidMLInference/blob/main/functionality.md): this file contains extensive descriptions of the functions, optional arguments, etc. 
