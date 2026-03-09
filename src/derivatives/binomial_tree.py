import numpy as np

class BinomialTreeEngine:
    """
    Prices options using the Cox-Ross-Rubinstein Binomial Tree model.
    Optimized for early-exercise American options.
    """
    def __init__(self, S, K, T, r, sigma, q, option_type='call', steps=300):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.option_type = option_type.lower()
        self.steps = int(steps)
        
    def price_tree(self):
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        
        # Risk-neutral probability incorporating continuous dividend yield
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity (Vectorized)
        prices = self.S * (u ** np.arange(self.steps, -1, -1)) * (d ** np.arange(0, self.steps + 1, 1))
        
        # Initialize option values at maturity
        if self.option_type == 'call':
            option_values = np.maximum(0, prices - self.K)
        else:
            option_values = np.maximum(0, self.K - prices)
            
        discount = np.exp(-self.r * dt)
        
        # Step backward through the tree
        for j in range(self.steps - 1, -1, -1):
            prices = prices[:-1] / u  # Update prices for the current step
            # Calculate continuation value
            continuation_value = discount * (p * option_values[:-1] + (1 - p) * option_values[1:])
            
            # Check for early exercise (American property)
            if self.option_type == 'call':
                exercise_value = np.maximum(0, prices - self.K)
            else:
                exercise_value = np.maximum(0, self.K - prices)
                
            # American options: Take the max of continuation or early exercise
            option_values = np.maximum(continuation_value, exercise_value)
            
        return option_values[0]

    def delta(self):
        """Standard tree delta calculation at node 0"""
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        
        prices_1 = np.array([self.S * u, self.S * d])
        
        if self.option_type == 'call':
            opt_1 = np.maximum(0, prices_1 - self.K)
        else:
            opt_1 = np.maximum(0, self.K - prices_1)
            
        discount = np.exp(-self.r * dt)
        # Simplified one-step look forward for rough delta
        # A true numerical greek engine overlay is preferred for consistency across the app
        # We will use Central Finite Difference in NumericalGreeksEngine to match everything else
        pass
