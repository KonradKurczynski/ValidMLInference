# ValidMLInference
 This repository hosts the code for the **ValidMLInference** package, implementing bias corrction methods described in [Battaglia, Christensen, Hansen & Sacher (2024)](https://cowles.yale.edu/research/cfdp-2421-inference-regression-variables-generated-ai-or-machine-learning). The two core functions are: 

 ## ols_bca
This procedure first computes the standard OLS estimator on a potentially mismeasured design matrix (Xhat), and then applies an additive correction based on a bias correction factor (fpr) that typically represents the false-positive rate estimated from a validation sample. The method also adjusts the variance estimator with a finite-sample correction term (m) to account for the uncertainty in the bias estimation.

    Parameters
    ----------
    Y : array_like, shape (n,)
        Response variable vector.
    Xhat : array_like, shape (n, d)
        Design matrix that may include measurement error due to AI/ML-generated regressors.
    fpr : float
        Bias correction factor (e.g., the false positive rate of misclassification) used to adjust 
        the OLS estimates.
    m : int or float
        Finite-sample adjustment parameter that reflects the effective size of the validation sample 
        or the degree of measurement error, used to correct the variance estimate.

    Returns
    -------
    b : ndarray, shape (d,)
        Bias-corrected regression coefficient estimates.
    V : ndarray, shape (d, d)
        Adjusted variance-covariance matrix for the bias-corrected estimator.


 ## one_step_unlabeled
