# ValidMLInference
 This repository hosts the code for the **ValidMLInference** package, implementing bias corrction methods described in [Battaglia, Christensen, Hansen & Sacher (2024)](https://arxiv.org/abs/2402.15585). <mark> A sample application of this package can be found in the file [example 1.ipynb](https://github.com/KonradKurczynski/ValidMLInference/blob/main/example%201.ipynb). 

 ## Getting Started 
This package can be installed with any default package manager, for instance, by typing
``` > pip install ValidMLInference ```  into the terminal. The core functions of the package are:

 ## ols_bca
This procedure first computes the standard OLS estimator on a design matrix (Xhat), the first column of which contains AI/ML-generated binary labels, and then applies an additive correction based on an estimate (fpr) of the false-positive rate computed externally. The method also adjusts the variance estimator with a finite-sample correction term to account for the uncertainty in the bias estimation.

    Parameters
    ----------
    Y : array_like, shape (n,)
        Response variable vector.
    Xhat : array_like, shape (n, d)
        Design matrix, the first column of which contains the AI/ML-generated binary covariates.
    fpr : float
        False positive rate of misclassification, used to correct the OLS estimates.
    m : int or float
        Size of the external sample used to estimate the classifier's false-positive rate. Can be set to 'inf' when the false-positive rate is known exactly.
    intercept : bool
        True by default, adds an intercept term to the estimated linear model. 

    Returns
    -------
    b : ndarray, shape (d,)
        Bias-corrected regression coefficient estimates.
    V : ndarray, shape (d, d)
        Adjusted variance-covariance matrix for the bias-corrected estimator.

 ## ols_bcm
This procedure first computes the standard OLS estimator on a design matrix (Xhat), the first column of which contains AI/ML-generated binary labels, and then applies a multiplicative correction based on an estimate (fpr) of the false-positive rate computed externally. The method also adjusts the variance estimator with a finite-sample correction term to account for the uncertainty in the bias estimation.

    Parameters
    ----------
    Y : array_like, shape (n,)
        Response variable vector.
    Xhat : array_like, shape (n, d)
        Design matrix, the first column of which contains the AI/ML-generated binary covariates.
    fpr : float
        False positive rate of misclassification, used to correct the OLS estimates.
    m : int or float
        Size of the external sample used to estimate the classifier's false-positive rate. Can be set to 'inf' when the false-positive rate is known exactly.
    intercept : bool
        True by default, adds an intercept term to the estimated linear model. 

    Returns
    -------
    b : ndarray, shape (d,)
        Bias-corrected regression coefficient estimates.
    V : ndarray, shape (d, d)
        Adjusted variance-covariance matrix for the bias-corrected estimator.



 ## one_step_unlabeled

This method jointly estimates the upstream (measurement) and downstream (regression) models using only the unlabeled likelihood. Leveraging JAX for automatic differentiation and optimization, it minimizes the negative log-likelihood to obtain the regression coefficients. The variance is then approximated via the inverse Hessian at the optimum.

    Parameters
    ----------
    Y : array_like, shape (n,)
        Response variable vector.
    Xhat : array_like, shape (n, d)
        Design matrix constructed from AI/ML-generated regressors.
    homoskedastic : bool, optional (default: False)
        If True, assumes a common error variance; otherwise, separate error variances are estimated.
    distribution : allows to specify the distribution of error terms, optional. By default, it's Normal(0,1).
        A custom distribution can be passed down as any jax-compatible PDF function that takes inputs (x, loc, scale).
    intercept : bool
        True by default, adds an intercept term to the estimated linear model. 

    Returns
    -------
    b : ndarray, shape (d,)
        Estimated regression coefficients extracted from the optimized parameter vector.
    V : ndarray, shape (d, d)
        Estimated variance-covariance matrix for the regression coefficients, computed as the inverse 
        of the Hessian of the objective function.


# ValidMLInference: example 1

```python
from ValidMLInference import ols, ols_bca, ols_bcm, one_step
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
```

### Parameters for simulation


```python
nsim    = 1000
n       = 16000      # training size
m       = 1000       # test size
p       = 0.05       # P(X=1)
kappa   = 1.0        # measurement‐error strength
fpr     = kappa / sqrt(n)

β0, β1       = 10.0, 1.0
σ0, σ1       = 0.3, 0.5

# Bayesian parameters for the false positive rate for BCA and BCM bias correction
α = [0.0, 0.5, 0.5]
β = [0.0, 2.0, 4.0]

# pre­allocate storage: (sim × 9 methods × 2 coefficients)
B = np.zeros((nsim, 9, 2))
S = np.zeros((nsim, 9, 2))
```

### Data Generation


```python
def generate_data(n, m, p, fpr, β0, β1, σ0, σ1):
    """
    Generates simulated data.

    Parameters:
      n, m: Python integers (number of training and test samples)
      p, p1: floats
      beta0, beta1: floats

    Returns:
      A tuple: ((train_Y, train_X), (test_Y, test_Xhat, test_X))
      where train_X and test_Xhat include a constant term as the second column.
    """
    N = n + m
    X    = np.zeros(N)
    Xhat = np.zeros(N)
    u    = np.random.rand(N)

    for j in range(N):
        if   u[j] <= fpr:
            X[j] = 1.0
        elif u[j] <= 2*fpr:
            Xhat[j] = 1.0
        elif u[j] <= p + fpr:
            X[j] = 1.0
            Xhat[j] = 1.0

    eps = np.random.randn(N)
    Y   = β0 + β1*X + (σ1*X + σ0*(1.0 - X))*eps

    # split into train vs test
    train_Y   = Y[:n]
    test_Y    = Y[n:]

    train_X   = Xhat[:n].reshape(-1, 1)
    test_Xhat = Xhat[n:].reshape(-1, 1)
    test_X    = X[n:].reshape(-1, 1)

    return (train_Y, train_X), (test_Y, test_Xhat, test_X)
```

### Bias-correction stage


```python
def update_results(B, S, b, V, i, method_idx):
    """
    Store coefficient estimates and their SEs into B and S.
    B,S have shape (nsim, nmethods, max_n_coefs).
    b is length d <= max_n_coefs.  V is d×d.
    """
    d = b.shape[0]
    for j in range(d):
        B[i, method_idx, j] = b[j]
        S[i, method_idx, j] = np.sqrt(max(V[j, j], 0.0))

for i in range(nsim):
    (tY, tX), (eY, eXhat, eX) = generate_data(
        n, m, p, fpr, β0, β1, σ0, σ1
    )

    # 1) OLS on unlabeled (Xhat)
    b, V, _ = ols(tY, tX, intercept = True)
    update_results(B, S, b, V, i, 0)

    # 2) OLS on labeled (true X)
    b, V, _ = ols(eY, eX, intercept = True)
    update_results(B, S, b, V, i, 1)

    # 3–8) Additive & multiplicative bias corrections
    fpr_hat = np.mean(eXhat[:,0] * (1.0 - eX[:,0]))
    for j in range(3):
        fpr_bayes = (fpr_hat*m + α[j]) / (m + α[j] + β[j])
        b, V = ols_bca(tY, tX, fpr_bayes, m)
        update_results(B, S, b, V, i, 2 + j)
        b, V = ols_bcm(tY, tX, fpr_bayes, m)
        update_results(B, S, b, V, i, 5 + j)

    # 9) One‐step unlabeled‐only
    b, V = one_step(tY, tX)
    update_results(B, S, b, V, i, 8)

    if (i+1) % 100 == 0:
        print(f"Done {i+1}/{nsim} sims")

```

    Done 100/1000 sims
    Done 200/1000 sims
    Done 300/1000 sims
    Done 400/1000 sims
    Done 500/1000 sims
    Done 600/1000 sims
    Done 700/1000 sims
    Done 800/1000 sims
    Done 900/1000 sims
    Done 1000/1000 sims


### Creating a Coverage Table


```python
def coverage(bgrid, b, se):
    """
    Computes the coverage probability for a grid of β values.

    For each value in bgrid, it computes the fraction of estimates b that
    lie within 1.96*se of that value.
    """
    cvg = np.empty_like(bgrid)
    for i, val in enumerate(bgrid):
        cvg[i] = np.mean(np.abs(b - val) <= 1.96 * se)
    return cvg
```


```python
true_beta1 = 1.0

methods = {
    "OLS θ̂":  0,
    "OLS θ": 1,
    "BCA‑0": 2,
    "BCA‑1": 3,
    "BCA‑2": 4,
    "BCM‑0": 5,
    "BCM‑1": 6,
    "BCM‑2": 7,
    "OSU":    8,
}

cov_dict = {}
for name, col in methods.items():
    slopes = B[:, col, 0]
    ses   = S[:, col, 0]
    # fraction of sims whose 95% CI covers true_beta1
    cov_dict[name] = np.mean(np.abs(slopes - true_beta1) <= 1.96 * ses)

cov_series = pd.Series(cov_dict, name=f"Coverage @ β₁={true_beta1}")
cov_series
```




    OLS θ̂    0.000
    OLS θ     0.941
    BCA‑0     0.878
    BCA‑1     0.909
    BCA‑2     0.907
    BCM‑0     0.887
    BCM‑1     0.906
    BCM‑2     0.908
    OSU       0.955
    Name: Coverage @ β₁=1.0, dtype: float64



### Recovering Coefficients and Standard Errors

Recall that the dataframe B stores our coefficient results while the dataframe S stores our standard errors. We can summarize our simulation results by averaging over the columns which store the results for the different simulation methods.


```python
nsim, nmethods, ncoeff = B.shape

method_names = [
    "OLS (θ̂)",
    "OLS (θ)",
    "BCA (j=0)",
    "BCA (j=1)",
    "BCA (j=2)",
    "BCM (j=0)",
    "BCM (j=1)",
    "BCM (j=2)",
    "1-Step"
]

results = []

for i in range(nmethods):
    row = {"Method": method_names[i]}
    
    for j, coef in enumerate(["Beta1", "Beta0"]):
        estimates = B[:, i, j]
        ses = S[:, i, j]
        mean_est = np.nanmean(estimates)
        mean_se = np.nanmean(ses)
        lower = np.percentile(estimates, 2.5)
        upper = np.percentile(estimates, 97.5)
        
        row[f"Est({coef})"] = f"{mean_est:.3f}"
        row[f"SE({coef})"] = f"{mean_se:.3f}"
        row[f"95% CI ({coef})"] = f"[{lower:.3f}, {upper:.3f}]"
    
    results.append(row)

df_results = pd.DataFrame(results).set_index("Method")
print(df_results)
```

              Est(Beta1) SE(Beta1)  95% CI (Beta1) Est(Beta0) SE(Beta0)  \
    Method                                                                
    OLS (θ̂)       0.833     0.021  [0.791, 0.871]     10.008     0.003   
    OLS (θ)        1.000     0.071  [0.858, 1.136]     10.000     0.010   
    BCA (j=0)      0.971     0.062  [0.871, 1.081]     10.001     0.004   
    BCA (j=1)      0.979     0.064  [0.880, 1.090]     10.001     0.004   
    BCA (j=2)      0.979     0.064  [0.880, 1.089]     10.001     0.004   
    BCM (j=0)      1.003     0.064  [0.877, 1.170]     10.000     0.004   
    BCM (j=1)      1.016     0.067  [0.887, 1.186]      9.999     0.004   
    BCM (j=2)      1.015     0.067  [0.887, 1.185]      9.999     0.004   
    1-Step         0.998     0.031  [0.930, 1.054]     10.000     0.003   
    
                 95% CI (Beta0)  
    Method                       
    OLS (θ̂)   [10.003, 10.013]  
    OLS (θ)     [9.982, 10.018]  
    BCA (j=0)   [9.994, 10.008]  
    BCA (j=1)   [9.994, 10.008]  
    BCA (j=2)   [9.994, 10.008]  
    BCM (j=0)   [9.990, 10.008]  
    BCM (j=1)   [9.990, 10.007]  
    BCM (j=2)   [9.990, 10.007]  
    1-Step      [9.995, 10.005]  

