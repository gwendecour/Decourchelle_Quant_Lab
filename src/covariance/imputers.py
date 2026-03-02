import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Requires scikit-learn
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Explicitly enable it
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

class BaseImputer(ABC):
    """
    Abstract Base Class for all imputation algorithms.
    Forces child classes to implement fit and transform methods.
    """
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method to fit and transform in one step."""
        self.fit(df)
        return self.transform(df)


class ForwardFillImputer(BaseImputer):
    """
    The baseline: Forward Fill. 
    It prolongs the last known price forward. Often used by default but dangerous as
    it artificially reduces volatility and skews correlation.
    """
    def fit(self, df: pd.DataFrame):
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # We also backfill just in case the first element itself is NaN
        return df.ffill().bfill()


class KNNImputerModel(BaseImputer):
    """
    K-Nearest Neighbors based Imputer.
    Finds cross-sectional neighbors to infer missing prices.
    """
    def __init__(self, n_neighbors: int = 5):
        super().__init__()
        self.imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')

    def fit(self, df: pd.DataFrame):
        self.imputer.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        imputed_data = self.imputer.transform(df)
        return pd.DataFrame(imputed_data, index=df.index, columns=df.columns)


from sklearn.ensemble import ExtraTreesRegressor

class MICEImputerModel(BaseImputer):
    """
    Multivariate Imputation by Chained Equations (MICE).
    Iteratively regressions missing variables against all other variables.
    Using ExtraTreesRegressor allows it to capture non-linear market regimes.
    """
    def __init__(self, max_iter: int = 10, random_state: int = 42):
        super().__init__()
        self.imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=10, random_state=random_state), 
            max_iter=max_iter, 
            random_state=random_state,
            tol=1e-3,
            initial_strategy='mean'
        )
        
    def fit(self, df: pd.DataFrame):
        self.imputer.fit(df)
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        imputed_data = self.imputer.transform(df)
        return pd.DataFrame(imputed_data, index=df.index, columns=df.columns)


class SVDImputerModel(BaseImputer):
    """
    Singular Value Decomposition (SVD) Imputer (Matrix Completion).
    Assumes financial returns can be explained by a few latent factors.
    Iteratively thresholded SVD to reconstruct the matrix.
    """
    def __init__(self, rank: int = None, max_iter: int = 50, tol: float = 1e-3):
        super().__init__()
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, df: pd.DataFrame):
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.values.copy()
        mask = np.isnan(data)
        
        # 1. Scale the data (SVD is highly sensitive to the scale of features)
        col_means = np.nanmean(data, axis=0)
        col_stds = np.nanstd(data, axis=0)
        # Prevent division by zero
        col_stds[col_stds == 0] = 1.0 
        
        scaled_data = (data - col_means) / col_stds
        
        # Initialize missing values in the scaled space with 0 (since it's standardized, mean=0)
        inds = np.where(mask)
        scaled_data[inds] = 0.0
        
        n_features = scaled_data.shape[1]
        k = self.rank if self.rank is not None else max(1, n_features // 2)
        
        for _ in range(self.max_iter):
            old_data = scaled_data.copy()
            
            # Perform SVD
            U, S, Vt = np.linalg.svd(scaled_data, full_matrices=False)
            
            # Low rank approximation
            S_k = np.zeros_like(S)
            S_k[:k] = S[:k]
            
            # Reconstruct (Still in scaled space)
            data_reconstructed = U @ np.diag(S_k) @ Vt
            
            # Update only the missing entries
            scaled_data[mask] = data_reconstructed[mask]
            
            # Check for convergence
            denom = np.linalg.norm(old_data[mask]) + 1e-8
            diff = np.linalg.norm(scaled_data[mask] - old_data[mask]) / denom
            if diff < self.tol:
                break
                
        # 2. De-scale the final imputed dataset back to the original price space
        final_data = (scaled_data * col_stds) + col_means
        
        # Ensure we only overwrite the NaN values to guard against floating point drifts
        data[mask] = final_data[mask]
        
        return pd.DataFrame(data, index=df.index, columns=df.columns)


class EMImputerModel(BaseImputer):
    """
    Expectation-Maximization (EM) Imputer for Multivariate Normal structure.
    Iteratively estimates the Maximum-Likelihood Mean and Covariance vectors,
    then assigns missing values as their conditional expectations.
    """
    def __init__(self, max_iter: int = 50, tol: float = 1e-4):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.mu = None
        self.cov = None
        
    def fit(self, df: pd.DataFrame):
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.values.copy()
        n, p = X.shape
        mask = np.isnan(X)
        
        # Initialization
        col_means = np.nanmean(X, axis=0)
        col_means[np.isnan(col_means)] = 0
        
        X_filled = X.copy()
        inds = np.where(mask)
        X_filled[inds] = np.take(col_means, inds[1])
        
        mu = np.mean(X_filled, axis=0)
        cov = np.cov(X_filled, rowvar=False) + np.eye(p) * 1e-6 # Epsilon for stability
        
        for _ in range(self.max_iter):
            mu_old = mu.copy()
            cov_old = cov.copy()
            
            # E-Step: Impute conditional expectation of missing variables 
            for i in range(n):
                missing_idx = mask[i, :]
                obs_idx = ~missing_idx
                
                if not np.any(missing_idx):
                    continue
                    
                mu_obs = mu[obs_idx]
                mu_miss = mu[missing_idx]
                
                sum_obs_obs = cov[np.ix_(obs_idx, obs_idx)]
                sum_miss_obs = cov[np.ix_(missing_idx, obs_idx)]
                
                try:
                    inv_sum_obs_obs = np.linalg.inv(sum_obs_obs)
                except np.linalg.LinAlgError:
                    inv_sum_obs_obs = np.linalg.pinv(sum_obs_obs)
                    
                x_obs = X[i, obs_idx]
                
                # E[ X_miss | X_obs ]
                x_miss_cond = mu_miss + sum_miss_obs @ inv_sum_obs_obs @ (x_obs - mu_obs)
                X_filled[i, missing_idx] = x_miss_cond
                
            # M-Step: Re-estimate parameters from the fully completed dataset
            mu = np.mean(X_filled, axis=0)
            cov = np.cov(X_filled, rowvar=False) + np.eye(p) * 1e-6
            
            delta_mu = np.linalg.norm(mu - mu_old)
            delta_cov = np.linalg.norm(cov - cov_old)
            
            if delta_mu < self.tol and delta_cov < self.tol:
                break
                
        self.mu = mu
        self.cov = cov
        
        return pd.DataFrame(X_filled, index=df.index, columns=df.columns)
