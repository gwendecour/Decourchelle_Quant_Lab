import numpy as np
import pandas as pd
from scipy.optimize import minimize

class CovarianceMatrixEstimator:
    """
    Estimator designed to extract a robust and well-conditioned expected covariance 
    matrix from price time series, adding a small ridge regularization (epsilon) 
    to ensure positive-definite status for the optimizer.
    """
    def __init__(self, ann_factor: int = 252):
        self.ann_factor = ann_factor

    def estimate(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the annualized sample covariance matrix from daily prices.
        Regularizes slightly if matrix condition is poor.
        """
        returns = np.log(prices_df / prices_df.shift(1)).dropna()
        cov_matrix = returns.cov() * self.ann_factor
        
        eps = 1e-8
        np_cov = cov_matrix.values
        np_cov = np_cov + np.eye(np_cov.shape[0]) * eps
        
        return pd.DataFrame(np_cov, index=cov_matrix.index, columns=cov_matrix.columns)


class ErrorMetrics:
    """
    Functional class for calculating mathematical distances between datasets.
    """
    @staticmethod
    def frobenius_norm(cov_imputed: pd.DataFrame, cov_true: pd.DataFrame) -> float:
        """
        Calculates the Frobenius distance between two covariance matrices.
        Returns || Σ_imputed - Σ_true ||_F
        """
        diff = cov_imputed.values - cov_true.values
        return np.linalg.norm(diff, ord='fro')


class PortfolioOptimizer:
    """
    Class responsible for building Min-Variance Portfolios to show 
    the business impact of a bad covariance matrix.
    """
    def __init__(self):
        pass

    def min_variance_portfolio(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Calculates the Minimum Variance Portfolio weights:
        w_mvp = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
        
        Uses an optimizer to handle potential numeric instabilities 
        better than pure matrix inversion, and enforces long-only bounds.
        """
        n_assets = len(cov_matrix)
        initial_weights = np.ones(n_assets) / n_assets
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        # The constraint: weights must sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Objective function: 1/2 * w^T * Σ * w
        def objective_function(w, cov):
            return 0.5 * np.dot(w.T, np.dot(cov, w))
            
        result = minimize(
            objective_function,
            initial_weights,
            args=(cov_matrix.values,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            # Fallback to analytical solution (allowing shorting if bounds fail)
            try:
                inv_cov = np.linalg.inv(cov_matrix.values)
                ones = np.ones(n_assets)
                w = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
                return pd.Series(w, index=cov_matrix.columns)
            except np.linalg.LinAlgError:
                return pd.Series(initial_weights, index=cov_matrix.columns)
                
        return pd.Series(result.x, index=cov_matrix.columns)

    def max_diversification_portfolio(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Maximum Diversification Portfolio: Maximizes the Diversification Ratio (DR).
        DR = (w^T * sigma) / sqrt(w^T * cov * w)
        Which is mathematically equivalent to minimizing w^T * cov * w subject to w^T * sigma = 1,
        then normalizing the weights to sum to 1.
        """
        n_assets = len(cov_matrix)
        vols = np.sqrt(np.diag(cov_matrix.values))
        
        initial_weights = np.ones(n_assets) / n_assets
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        # We need to maximize DR, meaning minimize -DR
        def neg_dr_objective(w, cov, vols):
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            weighted_vols = np.dot(w, vols)
            return - (weighted_vols / port_vol)
            
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        result = minimize(
            neg_dr_objective,
            initial_weights,
            args=(cov_matrix.values, vols),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            tol=1e-8
        )
        
        if not result.success:
            # Fallback to naive equal weight
            return pd.Series(initial_weights, index=cov_matrix.columns)
            
        weights = result.x / np.sum(result.x)
        return pd.Series(weights, index=cov_matrix.columns)

    def inverse_volatility_portfolio(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Naive Risk Parity: Weights inversely proportional to individual volatilities.
        """
        vols = np.sqrt(np.diag(cov_matrix.values))
        vols[vols < 1e-8] = 1e-8
        inv_vols = 1.0 / vols
        weights = inv_vols / np.sum(inv_vols)
        return pd.Series(weights, index=cov_matrix.columns)

    def risk_parity_portfolio(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Equal Risk Contribution (ERC) Portfolio: Each asset contributes exactly the same amount to variance.
        """
        n_assets = len(cov_matrix)
        initial_weights = np.ones(n_assets) / n_assets
        bounds = tuple((0.001, 1.0) for _ in range(n_assets)) # strictly positive for ERC
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        def erc_objective(w, cov):
            port_var = np.dot(w.T, np.dot(cov, w))
            risk_contrib = w * np.dot(cov, w)
            target_rc = port_var / n_assets
            # Scale the objective massively to prevent SLSQP numerical underflow/early stopping
            return np.sum(np.square(risk_contrib - target_rc)) * 1e6
            
        result = minimize(
            erc_objective,
            initial_weights,
            args=(cov_matrix.values,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            tol=1e-9
        )
        
        if not result.success:
            # Fallback to inverse volatility if numeric solver fails
            return self.inverse_volatility_portfolio(cov_matrix)
            
        return pd.Series(result.x, index=cov_matrix.columns)

    def evaluate_risk(self, weights: pd.Series, true_cov_matrix: pd.DataFrame, estimated_cov_matrix: pd.DataFrame) -> dict:
        """
        Calculates Ex-Ante Risk (what the model *thought* it would get) vs
        Ex-Post Risk (what it *actually* gets in reality since True Cov is different).
        """
        w = weights.values
        
        # Estimated Variance = w^T * Σ_est * w
        estimated_var = np.dot(w.T, np.dot(estimated_cov_matrix.values, w))
        
        # True Variance = w^T * Σ_true * w
        true_var = np.dot(w.T, np.dot(true_cov_matrix.values, w))
        
        return {
            'estimated_volatility': np.sqrt(estimated_var),
            'true_volatility': np.sqrt(true_var),
            'volatility_underestimation': np.sqrt(true_var) - np.sqrt(estimated_var)
        }
