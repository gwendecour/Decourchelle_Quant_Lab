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

    def price_american_option(self, option_type="call"):
        paths = self.generate_paths()
        steps, num_sims = paths.shape
        steps -= 1 

        K = self.K
        r = self.r
        dt = self.dt
        
        is_barrier = hasattr(self, 'barrier')
        if is_barrier:
            option_type = getattr(self, 'option_type', option_type)
            start_idx = int((self.window_start / self.T) * steps) if getattr(self, 'T', self.T) > 0 else 0
            end_idx = int((self.window_end / self.T) * steps) + 1 if getattr(self, 'T', self.T) > 0 else steps + 1
            
            touched_at_step = np.zeros_like(paths, dtype=bool)
            if self.direction == "up":
                touched_at_step[start_idx:end_idx] = paths[start_idx:end_idx] >= self.barrier
            elif self.direction == "down":
                touched_at_step[start_idx:end_idx] = paths[start_idx:end_idx] <= self.barrier
                
            touched_up_to = np.logical_or.accumulate(touched_at_step, axis=0)
            touched_anytime = touched_up_to[-1]

        if option_type == "call":
            intrinsic = np.maximum(paths[-1] - K, 0)
        elif option_type == "put":
            intrinsic = np.maximum(K - paths[-1], 0)
        else:
            intrinsic = np.maximum(paths[-1] - K, 0) 
            
        if is_barrier:
            if getattr(self, 'knock_type', 'out') == "out":
                cash_flows = np.where(touched_anytime, 0, intrinsic)
            else: # knock in
                cash_flows = np.where(touched_anytime, intrinsic, 0)
        else:
            cash_flows = intrinsic
        
        discount_factor = np.exp(-r * dt)
    
        for t in range(steps - 1, 0, -1):
            cash_flows = cash_flows * discount_factor 

            if option_type == "call":
                exercise_values = np.maximum(paths[t] - K, 0)
            elif option_type == "put":
                exercise_values = np.maximum(K - paths[t], 0)
            else:
                exercise_values = np.maximum(paths[t] - K, 0)
                
            if is_barrier:
                if getattr(self, 'knock_type', 'out') == "out":
                    cash_flows = np.where(touched_up_to[t], 0, cash_flows)
                    can_exercise = ~touched_up_to[t]
                else: 
                    can_exercise = touched_up_to[t]
            else:
                can_exercise = np.ones(num_sims, dtype=bool)
            
            itm_and_exercisable = (exercise_values > 0) & can_exercise
        
            if np.any(itm_and_exercisable):
                X = paths[t][itm_and_exercisable]
                Y = cash_flows[itm_and_exercisable]
                
                try:
                    coefficients = np.polyfit(X, Y, 2)
                    continuation_values = np.polyval(coefficients, X)
                except np.linalg.LinAlgError:
                    continuation_values = np.zeros_like(X)
                
                exercise_early = exercise_values[itm_and_exercisable] > continuation_values
                
                update_indices = np.where(itm_and_exercisable)[0][exercise_early]
                cash_flows[update_indices] = exercise_values[update_indices]
                
        final_discounted_cash_flows = cash_flows * discount_factor
        return np.mean(final_discounted_cash_flows)