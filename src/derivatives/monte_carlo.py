import numpy as np
from src.derivatives.pricing_model import EuropeanOption

class MonteCarloEngine:
    
    def __init__(self, S, K, T, r, sigma, q, num_simulations=100, num_steps=252, seed=None):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        
        self.N = int(num_simulations) 
        self.M = int(num_steps)       
        self.dt = T / num_steps       
        self.seed = seed
        
    def generate_paths(self):
        
        A = np.zeros((self.M+1, self.N))
        
        A[0] = self.S
        
        if self.seed is not None:
            np.random.seed(self.seed)
        
        Z = np.random.standard_normal((self.M, self.N))
        drift = (self.r - self.q - 0.5*self.sigma**2)*self.dt
        diffusion_multiplicator = self.sigma*np.sqrt(self.dt)
        
        for i in range(1, self.M+1):
            A[i] = A[i-1] * np.exp(drift + diffusion_multiplicator * Z[i-1])
        
        return A
    
    
    def price_european_call(self):
        
        A = self.generate_paths()
        
        Final_payoff = []
        Final_prices = A[self.M]
        Final_payoff = np.maximum(Final_prices - self.K, 0)
        
        discounted_price = np.exp(-self.r*self.T)*sum(Final_payoff)/self.N
        
        return discounted_price
    
    def price_barrier_option(self, barrier_level, option_type='call'):
        
        paths = self.generate_paths()
        
        max_spot_prices = np.max(paths, axis=0)
        is_alive = max_spot_prices < barrier_level
        
        final_prices = paths[-1]
        
        if option_type == "call":
            without_barrier = np.maximum(final_prices - self.K, 0)
        else : 
            without_barrier = np.maximum(self.K - final_prices, 0)

        return np.exp(-self.r * self.T)*np.mean(without_barrier*is_alive)        
        
        
        
    
    
            
        
        