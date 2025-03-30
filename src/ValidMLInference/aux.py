import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import numdifftools as nd
import jax
import jax.numpy as jnp
from jax import grad, jit, hessian
from jaxopt import LBFGS

# OLS with additive bias correction 
def ols_bca(Y, Xhat, fpr, m):
    """
    OLS estimator with additive bias correction.
    
    Computes the OLS estimator on Xhat, then uses the matrix
         A = [1,0,…,0]^T[1,0,…,0]  (i.e. A[0,0]=1)
    to obtain Γ = inv(sXX)*A. The bias‐corrected estimate is
         b_bc = b + fpr * (Γ b)
    and its variance is adjusted with a finite–sample term.
    """
    Xhat = np.asarray(Xhat)
    d = Xhat.shape[1]
    _b, _V, sXX = ols(Y, Xhat)
    A = np.zeros((d, d))
    A[0, 0] = 1.0
    Gamma = np.linalg.solve(sXX, A)
    b = _b + fpr * (Gamma @ _b)
    I = np.eye(d)
    V = (I + fpr * Gamma) @ _V @ (I + fpr * Gamma).T + fpr * (1.0 - fpr) * (Gamma @ (_V + np.outer(b, b)) @ Gamma.T) / m
    return b, V

# One–step estimation using only unlabeled data using JAX
@jit
def one_step_unlabeled(Y, Xhat, homoskedastic=False, distribution=None):
    """
    One–step estimator based solely on the unlabeled likelihood (JAX version).

    Parameters
    ----------
    Y : array_like, shape (n,)
        Response variable vector.
    Xhat : array_like, shape (n, d)
        Design matrix constructed from AI/ML-generated regressors.
    homoskedastic : bool, optional (default: False)
        If True, assumes a common error variance; otherwise, separate error variances are estimated.
    distribution : callable, optional
        A function that computes the probability density of the distribution to be used in the likelihood.
        It should have the signature pdf(x, loc, scale). If None, a Normal(0, 1) density is used.
      
    Returns
    -------
    b : ndarray, shape (d,)
        Estimated regression coefficients extracted from the optimized parameter vector.
    V : ndarray, shape (d, d)
        Estimated variance-covariance matrix for the regression coefficients, computed as the inverse 
        of the Hessian of the objective function.
    """
    def objective(theta):
        return likelihood_unlabeled_jax(Y, Xhat, theta, homoskedastic, distribution)
    
    theta_init = jnp.array(get_starting_values_unlabeled_jax(Y, Xhat, homoskedastic))
    solver = LBFGS(fun=objective, maxiter=200, tol=1e-8)
    sol = solver.run(theta_init)
    theta_opt = sol.params
    H = hessian(objective)(theta_opt)
    d = jnp.asarray(Xhat).shape[1]
    b = theta_opt[:d]
    V = jnp.linalg.inv(H)[:d, :d]
    return b, V

# Helper functions

def likelihood_unlabeled_jax(Y, Xhat, theta, homoskedastic, distribution=None):
    """
    Negative log–likelihood for the unlabeled data (JAX version).

    Parameters
    ----------
    Y : (n,) array
        Response array.
    Xhat : (n,d) array
        Design matrix.
    theta : array_like
        Parameter vector.
    homoskedastic : bool
        Flag indicating whether to assume a common error variance.
    distribution : callable, optional
        A function that computes the probability density of the distribution to be used.
        It should have a signature pdf(x, loc, scale). If None, a Normal(0, 1) density is used.
      
    Returns
    -------
    Negative log–likelihood (scalar).
    """
    Y = jnp.ravel(Y)
    d = Xhat.shape[1]
    b, w00, w01, w10, sigma0, sigma1 = theta_to_pars_jax(theta, d, homoskedastic)
    # Compute w11 from the raw parameters
    w11 = 1.0 / (1.0 + jnp.exp(theta[d]) + jnp.exp(theta[d+1]) + jnp.exp(theta[d+2]))
    mu = Xhat @ b  # (n,)
    
    # Choose the density function: default to normal_pdf if no custom distribution provided.
    pdf = normal_pdf if distribution is None else distribution

    # For each observation we have two cases depending on the first column of Xhat.
    # When Xhat[i,0] == 1:
    term1_1 = w11 * pdf(Y, mu, sigma1)
    term2_1 = w10 * pdf(Y, mu - b[0], sigma0)
    # When Xhat[i,0] == 0:
    term1_0 = w01 * pdf(Y, mu + b[0], sigma1)
    term2_0 = w00 * pdf(Y, mu, sigma0)
    indicator = Xhat[:, 0]
    # Use jnp.where to select the correct mixture for each observation.
    log_term = jnp.where(indicator == 1.0,
                         jnp.log(term1_1 + term2_1),
                         jnp.log(term1_0 + term2_0))
    return -jnp.sum(log_term)

def theta_to_pars_jax(theta, d, homoskedastic):
    """
    Transforms the parameter vector theta into interpretable parameters.
    
    Parameters
    ----------
    theta : 1D array
        Raw parameter vector.
    d : int
        Number of coefficients in b.
    homoskedastic : bool
        If True, use a single sigma.
      
    Returns
    -------
    b, w00, w01, w10, sigma0, sigma1
    """
    b = theta[:d]
    v = theta[d:d+3]
    exp_v = jnp.exp(v)
    w = exp_v / (1.0 + jnp.sum(exp_v))
    sigma0 = jnp.exp(theta[d+3])
    sigma1 = sigma0 if homoskedastic else jnp.exp(theta[d+4])
    return b, w[0], w[1], w[2], sigma0, sigma1

def get_starting_values_unlabeled_jax(Y, Xhat, homoskedastic):
    """
    Computes starting values based solely on the unlabeled data (JAX version).
    
    Parameters
    ----------
    Y : array_like
        Response vector.
    Xhat : array_like
        Design matrix.
    homoskedastic : bool
        Flag indicating whether to assume a common error variance.
      
    Returns
    -------
    A 1D JAX array with initial parameter estimates.
    """
    Y = jnp.ravel(Y)
    Xhat = jnp.asarray(Xhat)
    # Obtain an OLS estimate for b.
    b = ols_jax(Y, Xhat, se=False)
    u = Y - Xhat @ b
    sigma = jnp.std(u)
    # Define a helper pdf
    def pdf_func(y, loc, scale):
        return jnp.exp(-0.5 * jnp.square((y - loc) / scale)) / (jnp.sqrt(2 * jnp.pi) * scale)
    mu = Xhat @ b
    # For each observation, “impute” the missing true X based on comparing densities.
    cond1 = pdf_func(Y, mu, sigma) > pdf_func(Y, mu - b[0], sigma)
    cond2 = pdf_func(Y, mu + b[0], sigma) > pdf_func(Y, mu, sigma)
    X_imputed = jnp.where(Xhat[:, 0] == 1.0,
                          cond1.astype(jnp.float32),
                          cond2.astype(jnp.float32))
    freq00 = jnp.mean(((Xhat[:, 0] == 0.0) & (X_imputed == 0.0)).astype(jnp.float32))
    freq01 = jnp.mean(((Xhat[:, 0] == 0.0) & (X_imputed == 1.0)).astype(jnp.float32))
    freq10 = jnp.mean(((Xhat[:, 0] == 1.0) & (X_imputed == 0.0)).astype(jnp.float32))
    freq11 = jnp.mean(((Xhat[:, 0] == 1.0) & (X_imputed == 1.0)).astype(jnp.float32))
    w00 = jnp.maximum(freq00, 0.001)
    w01 = jnp.maximum(freq01, 0.001)
    w10 = jnp.maximum(freq10, 0.001)
    w11 = jnp.maximum(freq11, 0.001)
    w = jnp.array([w00, w01, w10, w11])
    w = w / jnp.sum(w)
    v = jnp.log(w[:3] / w[3])
    # Compute sigma0 and sigma1 over the two imputed groups
    mask0 = (X_imputed == 0.0)
    mask1 = (X_imputed == 1.0)
    sigma0 = subset_std(u, mask0)
    sigma1 = subset_std(u, mask1)
    sigma0 = jnp.where(jnp.isnan(sigma0), sigma1, sigma0)
    sigma1 = jnp.where(jnp.isnan(sigma1), sigma0, sigma1)
    if homoskedastic:
        p_val = jnp.mean(X_imputed)
        sigma_comb = sigma1 * p_val + sigma0 * (1.0 - p_val)
        return jnp.concatenate([b, v, jnp.array([jnp.log(sigma_comb)])])
    else:
        return jnp.concatenate([b, v, jnp.array([jnp.log(sigma0), jnp.log(sigma1)])])

def ols(Y, X, se=True):
    """
    OLS estimator with a (1/n)-scaled design matrix product.
    
    Computes
        sXX = (1/n) X'X   and   sXY = (1/n) X'Y,
        b = sXX^{-1} sXY.
    If se==True, it also computes a heteroskedastic‐consistent variance:
        Ω = sum_i u_i² (x_i x_i')
        V = inv(sXX) Ω inv(sXX) / n².
    """
    Y = np.asarray(Y).flatten()
    X = np.asarray(X)
    n, d = X.shape
    sXX = (1.0 / n) * (X.T @ X)
    sXY = (1.0 / n) * (X.T @ Y)
    b = np.linalg.solve(sXX, sXY)
    if se:
        Omega = np.zeros((d, d))
        for i in range(n):
            x_i = X[i, :]
            u = Y[i] - np.dot(x_i, b)
            Omega += (u**2) * np.outer(x_i, x_i)
        inv_sXX = np.linalg.inv(sXX)
        V = inv_sXX @ Omega @ inv_sXX / (n**2)
        return b, V, sXX
    else:
        return b
    
def ols_jax(Y, X, se=True):
    """
    Ordinary Least Squares estimator.

    Parameters
    ----------
    Y : (n,) array
        Response vector.
    X : (n,d) array
        Design matrix.
    se : bool, optional
        Whether to compute standard errors using a heteroskedastic–consistent formula.
      
    Returns
    -------
    b [, V, sXX]: b is the OLS coefficient; if se==True, V is the variance-covariance matrix.
    """
    Y = jnp.ravel(Y)
    X = jnp.asarray(X)
    n, d = X.shape
    sXX = (1.0 / n) * (X.T @ X)
    sXY = (1.0 / n) * (X.T @ Y)
    b = jnp.linalg.solve(sXX, sXY)
    if se:
        # Compute residuals
        residuals = Y - X @ b
        # Compute Omega = sum_i [u_i^2 * (x_i x_i^T)]
        Omega = jnp.sum(jnp.einsum('ni,nj->nij', X, X) * (residuals**2)[:, None, None], axis=0)
        inv_sXX = jnp.linalg.inv(sXX)
        V = inv_sXX @ Omega @ inv_sXX / (n**2)
        return b, V, sXX
    else:
        return b
    
def ols_bcm(Y, Xhat, fpr, m):
    """
    OLS estimator with multiplicative bias correction.
    
    Here the bias‐corrected estimate is
         b_bc = inv(I - fpr*Γ) b,
    with a corresponding finite–sample adjustment to the variance.
    """
    Xhat = np.asarray(Xhat)
    d = Xhat.shape[1]
    _b, _V, sXX = ols(Y, Xhat)
    A = np.zeros((d, d))
    A[0, 0] = 1.0
    Gamma = np.linalg.solve(sXX, A)
    I = np.eye(d)
    b = np.linalg.inv(I - fpr * Gamma) @ _b
    V = np.linalg.inv(I - fpr * Gamma) @ _V @ np.linalg.inv(I - fpr * Gamma).T + \
        fpr * (1.0 - fpr) * (Gamma @ (_V + np.outer(b, b)) @ Gamma.T) / m
    return b, V

# Jax-compatible distribution functions    
def log_normal_pdf(x, loc, scale):
    """Log–density of a Normal distribution."""
    return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(scale) - 0.5 * jnp.square((x - loc) / scale)

def normal_pdf(x, loc, scale):
    """Density of a Normal distribution."""
    return jnp.exp(log_normal_pdf(x, loc, scale))

def subset_std(x, mask):
    """
    Compute standard deviation over the subset of x where mask is True.
    """
    mask = mask.astype(jnp.float32)
    mean_val = jnp.sum(x * mask) / jnp.sum(mask)
    var = jnp.sum(mask * jnp.square(x - mean_val)) / jnp.sum(mask)
    return jnp.sqrt(var)