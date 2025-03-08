# ValidMLInference
 This repository hosts the code for the **ValidMLInference** package, implementing bias corrction methods described in [Battaglia, Christensen, Hansen & Sacher (2024)](https://arxiv.org/abs/2402.15585). The two core functions are: 

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
        Size of the external sample used to estimate the classifier's false-positive rate. Can be set to `'inf'` when the false-positive rate is known exactly.

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
    distribution : optional
        Reserved for interface compatibility. This parameter is not actively used in this implementation.

    Returns
    -------
    b : ndarray, shape (d,)
        Estimated regression coefficients extracted from the optimized parameter vector.
    V : ndarray, shape (d, d)
        Estimated variance-covariance matrix for the regression coefficients, computed as the inverse 
        of the Hessian of the objective function.


