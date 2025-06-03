from patsy import dmatrices
import numpy as np
from importlib import resources
import jax.numpy as jnp
import pandas as pd
from .implementations import (
    RegressionResult, _ols_bca_core, _ols_bcm_core, _one_step_gaussian_mixture_core, 
    _reorder_intercept_first, _ols_core, _one_step_core_with_treatment_idx,
    ols_bca_topic as ols_bca_topic_impl, 
    ols_bcm_topic as ols_bcm_topic_impl)
from typing import Dict, Any

def ols(
    *,
    formula: str | None = None,
    data: pd.DataFrame | None = None,
    Y: np.ndarray | None = None,
    X: np.ndarray | None = None,
    se: bool = True,
    intercept: bool = True, 
    names: list[str] | None = None,
) -> RegressionResult:  
    """
    Ordinary Least Squares regression.
    
    Returns
    -------
    result : RegressionResult
        Contains .coef, .vcov, and .names attributes
    """
    if formula is not None:
        if data is None:
            raise ValueError("`data` must be provided with a formula")
        
        if intercept:
            y, Xdf = dmatrices(formula, data, return_type="dataframe")
            intercept_already_in_matrix = True
        else:
            if '~ 0' not in formula and '~0' not in formula:
                formula_parts = formula.split('~')
                formula_no_intercept = formula_parts[0] + '~ 0 + ' + formula_parts[1]
            else:
                formula_no_intercept = formula
            y, Xdf = dmatrices(formula_no_intercept, data, return_type="dataframe")
            intercept_already_in_matrix = False
            
        Y = y.values.ravel()
        X = Xdf.values
        names = list(Xdf.columns)
        
    else:
        if Y is None or X is None:
            raise ValueError("must supply either formula+data or Y+X")
        Y = np.asarray(Y).ravel()
        X = np.asarray(X)

        if X.ndim == 1:
            X = X[:, None]

        if names is None:
            names = [f"x{i}" for i in range(1, X.shape[1] + 1)]
        intercept_already_in_matrix = False

    if intercept and not intercept_already_in_matrix:
        X = np.concatenate([X, np.ones((X.shape[0],1))], axis=1)
        if names is not None:
            names = names + ['Intercept']

    b, V, sXX = _ols_core(Y, X, se=se, intercept=False)  

    if intercept:
        if intercept_already_in_matrix:
            pass  
        else:
            b, V = _reorder_intercept_first(b, V, True)
            if names is not None:
                names = [names[-1]] + names[:-1]

    return RegressionResult(coef=b, vcov=V, names=names)

def ols_bca(
    *,
    formula: str | None = None,
    data: pd.DataFrame | None = None,
    Y: np.ndarray | None = None,
    Xhat: np.ndarray | None = None,
    fpr: float,
    m: int,
    intercept: bool = True,
    generated_var: str | None = None,      # ← new argument
    names: list[str] | None = None,
) -> RegressionResult:
    if formula is not None:
        if data is None:
            raise ValueError("`data` must be provided with a formula")
        if intercept:
            y, Xdf = dmatrices(formula, data, return_type="dataframe")
            intercept_in_matrix = 'Intercept' in Xdf.columns or '(Intercept)' in Xdf.columns
        else:
            lhs, rhs = formula.split('~', 1)
            no_int   = lhs + '~ 0 + ' + rhs if '~ 0' not in formula and '~0' not in formula else formula
            y, Xdf   = dmatrices(no_int, data, return_type="dataframe")
            intercept_in_matrix = False

        Y     = y.values.ravel()
        Xhat  = Xdf.values
        names = list(Xdf.columns)

    else:
        if Y is None or Xhat is None:
            raise ValueError("must supply either formula+data or Y+Xhat")
        Y    = np.asarray(Y).ravel()
        Xhat = np.asarray(Xhat)
        if Xhat.ndim == 1:
            Xhat = Xhat[:, None]
        intercept_in_matrix = False

        if names is None:
            names = [f"x{i}" for i in range(1, Xhat.shape[1] + 1)]

    if intercept and not intercept_in_matrix:
        Xhat = np.concatenate([Xhat, np.ones((Xhat.shape[0], 1))], axis=1)
        names = names + ['Intercept']
        intercept_in_matrix = True

    if generated_var is not None:
        if generated_var not in names:
            raise ValueError(f"Treatment variable '{generated_var}' not found. Available: {names}")
        target_idx = names.index(generated_var)
    else:
        if intercept_in_matrix:
            idx = names.index('Intercept') if 'Intercept' in names else names.index('(Intercept)')
            target_idx = 1 if idx == 0 else 0
        else:
            target_idx = 0

    b_corr, V_corr = _ols_bca_core(Y, Xhat, fpr=fpr, m=m, target_idx=target_idx)

    if intercept:
        b_corr, V_corr = _reorder_intercept_first(b_corr, V_corr, True)
        names         = [names[-1]] + names[:-1]

    return RegressionResult(coef=b_corr, vcov=V_corr, names=names)


def ols_bcm(
    *,
    formula: str | None = None,
    data: pd.DataFrame | None = None,
    Y: np.ndarray | None = None,
    Xhat: np.ndarray | None = None,
    fpr: float,
    m: int,
    intercept: bool = True,
    names: list[str] | None = None,
    generated_var: str | None = None,  
) -> RegressionResult:
    
    if formula is not None:
        if data is None:
            raise ValueError("`data` must be provided with a formula")
        
        if intercept:
            y, Xdf = dmatrices(formula, data, return_type="dataframe")
        else:
            if '~ 0' not in formula and '~0' not in formula:
                formula_parts = formula.split('~')
                formula_no_intercept = formula_parts[0] + '~ 0 + ' + formula_parts[1]
            else:
                formula_no_intercept = formula
            y, Xdf = dmatrices(formula_no_intercept, data, return_type="dataframe")
            
        Y = y.values.ravel()
        Xhat = Xdf.values
        names = list(Xdf.columns)
        
    else:
        if Y is None or Xhat is None:
            raise ValueError("must supply either formula+data or Y+Xhat")
        Y = np.asarray(Y).ravel()
        Xhat = np.asarray(Xhat)
        if Xhat.ndim == 1:
            Xhat = Xhat[:, None]
        
        if intercept:
            Xhat = np.concatenate([Xhat, np.ones((Xhat.shape[0], 1))], axis=1)
            if names is not None:
                names = names + ['Intercept']
            elif names is None:
                names = [f"x{i}" for i in range(1, Xhat.shape[1])] + ['Intercept']
        else:
            if names is None:
                names = [f"x{i}" for i in range(1, Xhat.shape[1] + 1)]

    if generated_var and generated_var in names:
        target_idx = names.index(generated_var)
    else:
        if 'Intercept' in names:
            intercept_pos = names.index('Intercept')
            target_idx = 1 if intercept_pos == 0 else 0
        else:
            target_idx = 0

    b_corr, V_corr = _ols_bcm_core(Y, Xhat, fpr=fpr, m=m, target_idx=target_idx)

    if intercept:
         b_corr, V_corr = _reorder_intercept_first(b_corr, V_corr, True)
         names         = [names[-1]] + names[:-1]
    
    return RegressionResult(coef=b_corr, vcov=V_corr, names=names)


def ols_bca_topic(
    *,
    Y: np.ndarray,
    Q: np.ndarray,
    W: np.ndarray,
    S: np.ndarray,
    B: np.ndarray,
    k: float,
    intercept: bool = True,
    names: list[str] | None = None,
) -> RegressionResult:
    """
    Bias-corrected additive estimator for topic model regression.
    
    Parameters
    ----------
    Y : np.ndarray
        Outcome variable
    Q : np.ndarray
        Covariates matrix
    W : np.ndarray
        Document-topic distribution matrix
    S : np.ndarray
        Topic-vocabulary matrix
    B : np.ndarray
        Vocabulary matrix
    k : float
        Bias correction parameter
    intercept : bool, default True
        Whether to include an intercept term
    names : list[str], optional
        Variable names
        
    Returns
    -------
    result : RegressionResult
        Contains .coef, .vcov, and .names attributes
    """
    Y = np.asarray(Y).ravel()
    Q = np.asarray(Q)
    W = np.asarray(W)
    S = np.asarray(S)
    B = np.asarray(B)
    
    if Q.ndim == 1:
        Q = Q[:, None]
    
    # Add intercept to Q if requested
    if intercept:
        Q = np.concatenate([Q, np.ones((Q.shape[0], 1))], axis=1)
    
    b, V = ols_bca_topic_impl(Y, Q, W, S, B, k)
    
    # Generate names if not provided
    if names is None:
        r = S.shape[0]  # number of topics
        q_vars = Q.shape[1] - (1 if intercept else 0)  # adjust for intercept
        topic_names = [f"topic_{i+1}" for i in range(r)]
        covar_names = [f"Q_{i+1}" for i in range(q_vars)]
        if intercept:
            covar_names = covar_names + ['Intercept']
        names = topic_names + covar_names
    
    # Reorder to put intercept first if present
    if intercept:
        b, V = _reorder_intercept_first(b, V, True)
        if names is not None:
            names = [names[-1]] + names[:-1]
    
    return RegressionResult(coef=b, vcov=V, names=names)


def ols_bcm_topic(
    *,
    Y: np.ndarray,
    Q: np.ndarray,
    W: np.ndarray,
    S: np.ndarray,
    B: np.ndarray,
    k: float,
    intercept: bool = True,
    names: list[str] | None = None,
) -> RegressionResult:
    """
    Bias-corrected multiplicative estimator for topic model regression.
    
    Parameters
    ----------
    Y : np.ndarray
        Outcome variable
    Q : np.ndarray
        Covariates matrix
    W : np.ndarray
        Document-topic distribution matrix
    S : np.ndarray
        Topic-vocabulary matrix
    B : np.ndarray
        Vocabulary matrix
    k : float
        Bias correction parameter
    intercept : bool, default True
        Whether to include an intercept term
    names : list[str], optional
        Variable names
        
    Returns
    -------
    result : RegressionResult
        Contains .coef, .vcov, and .names attributes
    """
    Y = np.asarray(Y).ravel()
    Q = np.asarray(Q)
    W = np.asarray(W)
    S = np.asarray(S)
    B = np.asarray(B)
    
    if Q.ndim == 1:
        Q = Q[:, None]
    
    # Add intercept to Q if requested
    if intercept:
        Q = np.concatenate([Q, np.ones((Q.shape[0], 1))], axis=1)
    
    b, V = ols_bcm_topic_impl(Y, Q, W, S, B, k)
    
    # Generate names if not provided
    if names is None:
        r = S.shape[0]  # number of topics
        q_vars = Q.shape[1] - (1 if intercept else 0)  # adjust for intercept
        topic_names = [f"topic_{i+1}" for i in range(r)]
        covar_names = [f"Q_{i+1}" for i in range(q_vars)]
        if intercept:
            covar_names = covar_names + ['Intercept']
        names = topic_names + covar_names
    
    # Reorder to put intercept first if present
    if intercept:
        b, V = _reorder_intercept_first(b, V, True)
        if names is not None:
            names = [names[-1]] + names[:-1]
    
    return RegressionResult(coef=b, vcov=V, names=names)


def one_step(
    *,
    formula: str | None = None,
    data: pd.DataFrame | None = None,
    Y: np.ndarray | None = None,
    Xhat: np.ndarray | None = None,
    generated_var: str | None = None,  
    homoskedastic: bool = False,
    distribution=None,
    intercept: bool = True,
    names: list[str] | None = None,
) -> RegressionResult:
    
    if formula is not None:
        if data is None:
            raise ValueError("`data` must be provided with a formula")
        
        if intercept:
            y, Xdf = dmatrices(formula, data, return_type="dataframe")
        else:
            if '~ 0' not in formula and '~0' not in formula:
                formula_parts = formula.split('~')
                formula_no_intercept = formula_parts[0] + '~ 0 + ' + formula_parts[1]
            else:
                formula_no_intercept = formula
            y, Xdf = dmatrices(formula_no_intercept, data, return_type="dataframe")
            
        Y = y.values.ravel()
        Xhat = Xdf.values
        names = list(Xdf.columns)
        

        if generated_var is None:
            if 'Intercept' in names:
                treatment_idx = next(i for i, name in enumerate(names) if name != 'Intercept')
            else:
                treatment_idx = 0
        else:
            if generated_var not in names:
                raise ValueError(f"Treatment variable '{generated_var}' not found in design matrix. Available: {names}")
            treatment_idx = names.index(generated_var)
        
    else:
        if Y is None or Xhat is None:
            raise ValueError("must supply either formula+data or Y+Xhat")
        Y = np.asarray(Y).ravel()
        Xhat = np.asarray(Xhat)
        if Xhat.ndim == 1:
            Xhat = Xhat[:, None]
        
        if generated_var is not None:
            if names is None:
                raise ValueError("When using generated_var with arrays, you must provide names")
            if generated_var not in names:
                raise ValueError(f"Treatment variable '{generated_var}' not found in names. Available: {names}")
            treatment_idx = names.index(generated_var)
        else:
            treatment_idx = 0
        
        if intercept:
            Xhat = np.concatenate([Xhat, np.ones((Xhat.shape[0], 1))], axis=1)
            if names is not None:
                names = names + ['Intercept']
            elif names is None:
                names = [f"x{i}" for i in range(1, Xhat.shape[1])] + ['Intercept']
        else:
            if names is None:
                names = [f"x{i}" for i in range(1, Xhat.shape[1] + 1)]

    treatment_col = Xhat[:, treatment_idx]
    unique_vals = np.unique(treatment_col)
    if not (len(unique_vals) == 2 and set(unique_vals) == {0.0, 1.0}):
        treatment_name = names[treatment_idx] if names else f"column {treatment_idx}"
        raise ValueError(f"Treatment variable '{treatment_name}' must be binary (0/1). Found values: {unique_vals}")

    b, V = _one_step_core_with_treatment_idx(Y, Xhat, treatment_idx, 
                                           homoskedastic=homoskedastic, 
                                           distribution=distribution)
    if intercept:
        b, V = _reorder_intercept_first(b, V, True)
        names         = [names[-1]] + names[:-1]
   
    return RegressionResult(coef=b, vcov=V, names=names)

def one_step_gaussian_mixture(
    *,
    formula: str | None = None,
    data: pd.DataFrame | None = None,
    Y: np.ndarray | None = None,
    Xhat: np.ndarray | None = None,
    generated_var: str | None = None,    # ← new!
    k: int = 2,
    homosked: bool = False,
    nguess: int = 10,
    maxiter: int = 100,
    seed: int = 0,
    intercept: bool = True,
    names: list[str] | None = None,
) -> RegressionResult:
    
    if formula is not None:
        if data is None:
            raise ValueError("`data` must be provided with a formula")
        if intercept:
            y, Xdf = dmatrices(formula, data, return_type="dataframe")
            intercept_in_matrix = 'Intercept' in Xdf.columns or '(Intercept)' in Xdf.columns
        else:
            lhs, rhs = formula.split('~', 1)
            no_int   = lhs + '~ 0 + ' + rhs if '~ 0' not in formula and '~0' not in formula else formula
            y, Xdf   = dmatrices(no_int, data, return_type="dataframe")
            intercept_in_matrix = False

        Y     = y.values.ravel()
        Xhat  = Xdf.values
        names = list(Xdf.columns)

    else:
        if Y is None or Xhat is None:
            raise ValueError("must supply either formula+data or Y+Xhat")
        Y    = np.asarray(Y).ravel()
        Xhat = np.asarray(Xhat)
        if Xhat.ndim == 1:
            Xhat = Xhat[:, None]
        intercept_in_matrix = False

        if names is None:
            names = [f"x{i}" for i in range(1, Xhat.shape[1] + 1)]

    if intercept and not intercept_in_matrix:
        Xhat = np.concatenate([Xhat, np.ones((Xhat.shape[0],1))], axis=1)
        names = names + ['Intercept']
        intercept_in_matrix = True

    if generated_var is not None:
        if generated_var not in names:
            raise ValueError(
                f"Treatment variable '{generated_var}' not found. Available: {names}"
            )
        tidx = names.index(generated_var)
        if tidx != 0:
            order = [tidx] + [i for i in range(len(names)) if i != tidx]
            Xhat = Xhat[:, order]
            names = [names[i] for i in order]

    b_jax, V_jax = _one_step_gaussian_mixture_core(
        Y, Xhat,
        k=k,
        homosked=homosked,
        nguess=nguess,
        maxiter=maxiter,
        seed=seed,
    )
    b = np.array(b_jax)
    V = np.array(V_jax)

    b, V = _reorder_intercept_first(b, V, True)
    names = [names[-1]] + names[:-1]

    return RegressionResult(coef=b, vcov=V, names=names)

def remote_work_data() -> pd.DataFrame:
    data_path = resources.files("ValidMLInference") / "data" / "remote_work_data.csv"
    return pd.read_csv(data_path)

def topic_model_data() -> Dict[str, Any]:
    data_path = resources.files("ValidMLInference") / "data" / "topic_model_data.npz"
    
    with resources.as_file(data_path) as file_path:
        with np.load(file_path) as data:
            return {
                'covars': data['covars'],
                'estimation_data': {'ly': data['estimation_data_ly']},
                'gamma_draws': data['gamma_draws'],
                'theta_est_full': data['theta_est_full'],
                'theta_est_samp': data['theta_est_samp'],
                'beta_est_full': data['beta_est_full'],
                'beta_est_samp': data['beta_est_samp'],
                'lda_data': data['lda_data']
            }