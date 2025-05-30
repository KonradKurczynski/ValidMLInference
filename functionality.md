# ols

Ordinary Least Squares regression with support for both formula and array interfaces. This function provides a unified interface for fitting linear models using either patsy formulas with pandas DataFrames or raw NumPy arrays.

**Usage Options**
----------
**Option 1: Formula Interface**
- **formula** : str - A patsy-style formula string (e.g., 'y ~ x1 + x2')
- **data** : pd.DataFrame - DataFrame containing the variables referenced in the formula

**Option 2: Array Interface**  
- **Y** : array_like, shape (n,) - Response variable vector
- **X** : array_like, shape (n, d) - Design matrix

**Additional Parameters**
----------

**se** : bool, optional (default: True)  
    Whether to compute standard errors using heteroskedastic-consistent estimator.

**intercept** : bool, optional (default: True)  
    Whether to include an intercept term in the model.

**names** : list[str], optional  
    Variable names for the coefficients. If not provided, default names are generated.

**Returns**
-------
**result** : RegressionResult  
    Object containing coefficient estimates (.coef), variance-covariance matrix (.vcov), and variable names (.names).

---

# ols_bca

Additive bias-corrected OLS estimator for models with AI/ML-generated binary covariates. This procedure first computes the standard OLS estimator on a design matrix, then applies an additive correction based on an estimate of the false-positive rate computed externally. The method also adjusts the variance estimator with a finite-sample correction term to account for the uncertainty in the bias estimation.

**Usage Options**
----------
**Option 1: Formula Interface**
- **formula** : str - A patsy-style formula string
- **data** : pd.DataFrame - DataFrame containing the variables referenced in the formula

**Option 2: Array Interface**  
- **Y** : array_like, shape (n,) - Response variable vector
- **Xhat** : array_like, shape (n, d) - Design matrix containing AI/ML-generated binary covariates

**Required Parameters**
----------

**fpr** : float  
    False positive rate of misclassification, used to correct the OLS estimates.

**m** : int  
    Size of the external sample used to estimate the classifier's false-positive rate. Can be set to a large number when the false-positive rate is known exactly.

**intercept** : bool, optional (default: True)  
    Whether to include an intercept term in the model.

**treatment_var** : str, optional  
    Name of the treatment variable to apply bias correction to. If not specified, defaults to the first non-intercept variable.

**names** : list[str], optional  
    Variable names for the coefficients. If not provided, default names are generated.

**Returns**
-------
**result** : RegressionResult  
    Object containing bias-corrected coefficient estimates (.coef), adjusted variance-covariance matrix (.vcov), and variable names (.names).

---

# ols_bcm

Multiplicative bias-corrected OLS estimator for models with AI/ML-generated binary covariates. This procedure first computes the standard OLS estimator on a design matrix, then applies a multiplicative correction based on an estimate of the false-positive rate computed externally. The method also adjusts the variance estimator with a finite-sample correction term to account for the uncertainty in the bias estimation.

**Additional Parameters**
----------
**formula** : str, optional  
    A patsy-style formula string. If provided, `data` must also be specified.

**data** : pd.DataFrame, optional  
    DataFrame containing the variables referenced in the formula.

**Y** : array_like, shape (n,), optional  
    Response variable vector. Required if not using formula interface.

**Xhat** : array_like, shape (n, d), optional  
    Design matrix containing AI/ML-generated binary covariates. Required if not using formula interface.

**fpr** : float  
    False positive rate of misclassification, used to correct the OLS estimates.

**m** : int  
    Size of the external sample used to estimate the classifier's false-positive rate. Can be set to a large number when the false-positive rate is known exactly.

**intercept** : bool, optional (default: True)  
    Whether to include an intercept term in the model.

**treatment_variable** : str, optional  
    Name of the treatment variable to apply bias correction to. If not specified, defaults to the first non-intercept variable.

**names** : list[str], optional  
    Variable names for the coefficients. If not provided, default names are generated.

**Returns**
-------
**result** : RegressionResult  
    Object containing bias-corrected coefficient estimates (.coef), adjusted variance-covariance matrix (.vcov), and variable names (.names).

---

# one_step

Joint estimation of upstream (measurement) and downstream (regression) models using only unlabeled data. This method leverages JAX for automatic differentiation and optimization to minimize the negative log-likelihood and obtain regression coefficients. The variance is approximated via the inverse Hessian at the optimum. This approach is particularly useful when true labels are unavailable but AI/ML-generated proxy labels exist.

**Additional Parameters**
----------
**formula** : str, optional  
    A patsy-style formula string. If provided, `data` must also be specified.

**data** : pd.DataFrame, optional  
    DataFrame containing the variables referenced in the formula.

**Y** : array_like, shape (n,), optional  
    Response variable vector. Required if not using formula interface.

**Xhat** : array_like, shape (n, d), optional  
    Design matrix constructed from AI/ML-generated regressors. Required if not using formula interface.

**treatment_var** : str, optional  
    Name of the binary treatment variable. If not specified, defaults to the first non-intercept variable.

**homoskedastic** : bool, optional (default: False)  
    If True, assumes a common error variance; otherwise, separate error variances are estimated for treatment and control groups.

**distribution** : callable, optional  
    Custom distribution for error terms. Must be a JAX-compatible PDF function with signature (x, loc, scale). Defaults to Normal(0,1).

**intercept** : bool, optional (default: True)  
    Whether to include an intercept term in the model.

**names** : list[str], optional  
    Variable names for the coefficients. If not provided, default names are generated.

**Returns**
-------
**result** : RegressionResult  
    Object containing estimated regression coefficients (.coef), variance-covariance matrix (.vcov), and variable names (.names).

---

# one_step_gaussian_mixture

Joint estimation using Gaussian mixture models for error terms. This extends the basic one-step estimator by allowing the error distribution to be a mixture of Gaussian components, providing greater flexibility in modeling heterogeneous populations or complex error structures.

**Usage Options**
----------
**Option 1: Formula Interface**
- **formula** : str - A patsy-style formula string
- **data** : pd.DataFrame - DataFrame containing the variables referenced in the formula

**Option 2: Array Interface**  
- **Y** : array_like, shape (n,) - Response variable vector
- **Xhat** : array_like, shape (n, d) - Design matrix constructed from AI/ML-generated regressors

**Additional Parameters**
----------

**treatment_var** : str, optional  
    Name of the binary treatment variable. If not specified, defaults to the first variable.

**k** : int, optional (default: 2)  
    Number of components in the Gaussian mixture model.

**homosked** : bool, optional (default: False)  
    If True, assumes common component variances across treatment groups.

**nguess** : int, optional (default: 10)  
    Number of random restarts for optimization to avoid local minima.

**maxiter** : int, optional (default: 100)  
    Maximum number of optimization iterations.

**seed** : int, optional (default: 0)  
    Random seed for reproducible results.

**intercept** : bool, optional (default: True)  
    Whether to include an intercept term in the model.

**names** : list[str], optional  
    Variable names for the coefficients. If not provided, default names are generated.

**Returns**
-------
**result** : RegressionResult  
    Object containing estimated regression coefficients (.coef), variance-covariance matrix (.vcov), and variable names (.names).

---

## load_dataset

Loads the built-in remote work dataset for demonstration and testing purposes.

**Returns**
-------
**data** : pd.DataFrame  
    DataFrame containing the remote work dataset with variables for analysis.
