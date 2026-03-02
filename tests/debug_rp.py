import sys
import pandas as pd
import numpy as np

sys.path.append("c:/Users/Gwendal/OneDrive/Bureau/Python project/Decourchelle_Quant_Lab/")
from src.covariance.evaluator import PortfolioOptimizer, CovarianceMatrixEstimator
from src.covariance.data_pipeline import MarketDataProvider, DataCorruptor
from src.covariance.imputers import ForwardFillImputer

def test_rp():
    # Load actual data
    tickers_list = ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'JNJ', 'V', 'PG')
    target_tickers = ('AMZN', 'META', 'TSLA')
    start = '2023-01-01'
    end = '2024-01-01'
    miss_rate = 0.3
    m_method = 'MCAR'
    
    # Get True Cov
    provider = MarketDataProvider(list(tickers_list), start, end)
    gt_df = provider.fetch_data()
    estimator = CovarianceMatrixEstimator(ann_factor=252)
    true_cov = estimator.estimate(gt_df)
    
    # Get FFill Cov
    corr = DataCorruptor(miss_rate, method=m_method, target_tickers=target_tickers).corrupt(gt_df, random_state=42)
    ff_filled = ForwardFillImputer().fit_transform(corr)
    ffill_cov = estimator.estimate(ff_filled)
    
    opt = PortfolioOptimizer()
    
    lines = []
    lines.append("\n--- RISK PARITY ---")
    w_gt = opt.risk_parity_portfolio(true_cov)
    w_ffill = opt.risk_parity_portfolio(ffill_cov)
    lines.append("True RP weights:\n" + str(w_gt.head(3)))
    lines.append("\nFFill RP weights:\n" + str(w_ffill.head(3)))
    diff = np.sum(np.abs(w_gt - w_ffill))
    lines.append(f"\nDifference RP: {diff}")
    
    lines.append("\n--- MAX DIVERSIFICATION ---")
    w_gt_md = opt.max_diversification_portfolio(true_cov)
    w_ffill_md = opt.max_diversification_portfolio(ffill_cov)
    lines.append("True MD weights:\n" + str(w_gt_md.head(3)))
    lines.append("\nFFill MD weights:\n" + str(w_ffill_md.head(3)))
    diff_md = np.sum(np.abs(w_gt_md - w_ffill_md))
    lines.append(f"\nDifference MD: {diff_md}")

    with open("c:/tmp/rp_test_results.txt", "w") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    test_rp()
