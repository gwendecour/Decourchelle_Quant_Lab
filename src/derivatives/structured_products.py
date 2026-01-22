import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from src.derivatives.monte_carlo import MonteCarloEngine
from src.derivatives.instruments import FinancialInstrument
import plotly.express as px

class PhoenixStructure(MonteCarloEngine, FinancialInstrument):
    
    def __init__(self, **kwargs):
        # 1. Récupération des paramètres
        S = float(kwargs.get('S'))
        self.nominal = S
        self.coupon_rate = kwargs.get('coupon_rate')
        
        # Gestion des barrières (entrée en % ou absolu, ici on standardise en absolu)
        self.autocall_barrier = S * kwargs.get('autocall_barrier')
        self.protection_barrier = S * kwargs.get('protection_barrier')
        self.coupon_barrier = S * kwargs.get('coupon_barrier')
        
        # Fréquence d'observation (Défaut à 4 = Trimestriel)
        self.obs_frequency = kwargs.get('obs_frequency', 4)
        
        maturity = float(kwargs.get('T'))
        
        # 2. Calcul du nombre de steps pour le Monte Carlo
        steps = max(int(252 * maturity), 1)
        self.steps = steps
        
        num_simulations = kwargs.get('num_simulations', 10000)
        self.num_simulations = num_simulations

        # 3. Init Moteur Monte Carlo (Parent 1)
        MonteCarloEngine.__init__(self, 
            S=S, K=S, T=maturity, 
            r=kwargs.get('r'), 
            sigma=kwargs.get('sigma'), 
            q=kwargs.get('q', 0.0), 
            num_simulations=num_simulations, 
            num_steps=steps, 
            seed=kwargs.get('seed')
        )
        
        # 4. Init Instrument (Parent 2)
        FinancialInstrument.__init__(self, **kwargs)

    def get_observation_indices(self):
        step_size = int(252 / self.obs_frequency)
        # On s'assure de ne pas dépasser le nombre de steps total
        indices = np.arange(step_size, self.steps + 1, step_size, dtype=int)
        return indices

    def calculate_payoffs_distribution(self):
        # Ta logique existante (Back-end)
        paths = self.generate_paths() 
        payoffs = np.zeros(self.N)
        active_paths = np.ones(self.N, dtype=bool)
        indices = self.get_observation_indices()
        
        coupon_amt = self.nominal * self.coupon_rate * (1.0/self.obs_frequency)
        
        for i, idx in enumerate(indices):
            if idx >= len(paths): break
            current_prices = paths[idx]
            
            # Conditions
            did_autocall = (current_prices >= self.autocall_barrier) & active_paths
            did_just_coupon = (current_prices >= self.coupon_barrier) & (current_prices < self.autocall_barrier) & active_paths
            
            # Discounting
            time_fraction = idx / 252.0
            df = np.exp(-self.r * time_fraction)
            
            # Payoff Logic
            payoffs[did_just_coupon] += coupon_amt * df
            payoffs[did_autocall] += (self.nominal + coupon_amt) * df
            
            active_paths[did_autocall] = False
            if not np.any(active_paths): break
    
        if np.any(active_paths):
            final_prices = paths[-1]
            survivors = active_paths
            df_final = np.exp(-self.r * self.T)
            
            safe_mask = survivors & (final_prices >= self.protection_barrier)
            payoffs[safe_mask] += self.nominal * df_final
            
            crash_mask = survivors & (final_prices < self.protection_barrier)
            payoffs[crash_mask] += final_prices[crash_mask] * df_final

        return payoffs

    def price(self):
        payoffs = self.calculate_payoffs_distribution()
        return np.mean(payoffs)

    def plot_payoff(self, spot_range=None):
        """
        Plots the Phoenix payoff at maturity.
        CORRECTED: Uses absolute barrier levels from __init__.
        """
        # 1. Recuperation des Barrieres (Déjà en absolu dans votre __init__)
        prot_level = self.protection_barrier 
        cpn_level = self.coupon_barrier
        
        # 2. Dynamic Range
        # On descend assez bas pour bien voir la barrière PDI
        low_bound = min(self.S * 0.3, prot_level * 0.8)
        high_bound = self.S * 1.5
        spots = np.linspace(low_bound, high_bound, 200)
        payoffs = []
        
        for s in spots:
            # Payoff Logic at Maturity
            if s >= cpn_level:
                # Scénario favorable : Capital + Coupon
                val = 1.0 + self.coupon_rate 
            elif s >= prot_level:
                # Scénario neutre : Capital protégé
                val = 1.0
            else:
                # Scénario perte : PDI (Put Down In)
                val = s / self.S
            
            payoffs.append(val * 100) # En pourcentage du nominal

        # 3. Plot
        fig = go.Figure()
        
        # Trace Payoff
        fig.add_trace(go.Scatter(
            x=spots, y=payoffs, 
            mode='lines', 
            name=' ', # ASTUCE : Espace vide pour éviter "undefined" au survol
            line=dict(color='#00CC96', width=3),
            hovertemplate="Spot: %{x:.2f}<br>Payoff: %{y:.1f}%<extra></extra>"
        ))

        # Barrières Verticales
        fig.add_vline(x=prot_level, line_dash="dash", line_color="red", 
                      annotation_text=f"Prot: {prot_level:.2f}")
        
        fig.add_vline(x=cpn_level, line_dash="dash", line_color="orange", 
                      annotation_text=f"Cpn: {cpn_level:.2f}", annotation_position="top")

        # Mise en page propre
        fig.update_layout(
            title=" ",
            xaxis_title="Spot Price at Maturity",
            yaxis_title="Payoff (% Nominal)",
            template="plotly_white",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode="x unified"
        )
        return fig

    def greeks(self):
        # Utilisation de la méthode des différences finies (Bump & Revalue)
        original_seed = self.seed if self.seed else 42
        self.seed = original_seed
        base_price = self.price()
        
        # Delta & Gamma
        epsilon = self.S * 0.01 
        orig_S = self.S
        
        self.S = orig_S + epsilon
        self.seed = original_seed
        p_up = self.price()
        
        self.S = orig_S - epsilon
        self.seed = original_seed
        p_down = self.price()
        
        self.S = orig_S # Reset
        
        delta = (p_up - p_down) / (2 * epsilon)
        gamma = (p_up - 2 * base_price + p_down) / (epsilon**2)
        
        # Vega
        orig_sigma = self.sigma
        self.sigma = orig_sigma + 0.01
        self.seed = original_seed
        p_vol_up = self.price()
        self.sigma = orig_sigma
        
        vega = p_vol_up - base_price
        
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": 0.0, "rho": 0.0}
    

    # ==========================================================================
    # NEW VISUALIZATIONS 
    # ==========================================================================

    def plot_phoenix_tunnel(self):
        """
        Visualise 200 chemins Monte Carlo colorés selon leur scénario final
        (Autocall = Vert, Maturité Safe = Gris, Perte = Rouge).
        """
        # On force 1000 simulations pour ce graph spécifique (rapide & lisible)
        original_N = self.N
        self.N = 1000 
        
        paths = self.generate_paths()
        obs_indices = self.get_observation_indices()
        
        # Logique de tri des chemins
        # Attention: paths est de dimension (Steps+1, Simulations)
        obs_prices = paths[obs_indices] # (Obs_Dates, Simulations)
        
        # Masque Autocall : Si à n'importe quelle date d'obs, Prix >= Barrière Autocall
        autocall_mask = np.any(obs_prices >= self.autocall_barrier, axis=0)
        
        final_prices = paths[-1]
        # Masque Crash : Pas autocallé ET fini sous la protection
        crash_mask = (~autocall_mask) & (final_prices < self.protection_barrier)
        # Masque Safe : Pas autocallé MAIS fini au dessus protection
        safe_mask = (~autocall_mask) & (final_prices >= self.protection_barrier)
        
        # Préparation Plotly
        fig = go.Figure()
        
        # Limite d'affichage pour ne pas surcharger le navigateur (200 lignes max)
        max_lines = 200
        x_axis = np.arange(paths.shape[0])
        
        # Fonction helper pour tracer des groupes de lignes
        def add_lines(mask, color, name, opacity):
            indices = np.where(mask)[0]
            if len(indices) == 0: return
            # On prend les 'max_lines' premiers chemins de ce groupe
            selected = indices[:max_lines]
            
            # Pour Plotly, tracer 100 lignes séparées est lourd. 
            # Astuce: On met tout dans une seule trace avec des 'None' entre les lignes
            x_flat = []
            y_flat = []
            for idx in selected:
                x_flat.extend(x_axis)
                x_flat.append(None) # Rupture de ligne
                y_flat.extend(paths[:, idx])
                y_flat.append(None)
            
            fig.add_trace(go.Scatter(
                x=x_flat, y=y_flat, 
                mode='lines', 
                line=dict(color=color, width=1), 
                opacity=opacity,
                name=name,
                showlegend=True
            ))

        # 1. Tracé des Chemins
        add_lines(autocall_mask, 'green', 'Autocall (Early Exit)', 0.15)
        add_lines(safe_mask, 'gray', 'Maturity (Capital Protected)', 0.4)
        add_lines(crash_mask, 'red', 'Loss (Barrier Hit)', 0.6)
        
        # 2. Barrières
        days = paths.shape[0] - 1
        fig.add_hline(y=self.autocall_barrier, line_dash="dash", line_color="green", annotation_text="Autocall Lvl")
        fig.add_hline(y=self.protection_barrier, line_dash="dash", line_color="red", annotation_text="Protection Lvl")
        if self.coupon_barrier != self.protection_barrier:
            fig.add_hline(y=self.coupon_barrier, line_dash="dot", line_color="cyan", annotation_text="Coupon Lvl")
            
        # 3. Dates d'observation (Lignes verticales)
        for idx in obs_indices:
            fig.add_vline(x=idx, line_width=1, line_color="white", opacity=0.2)
            
        # 4. Boite de Statistiques (Annotations)
        n_auto, n_safe, n_crash = np.sum(autocall_mask), np.sum(safe_mask), np.sum(crash_mask)
        stats_text = (
            f"<b>SCENARIOS (N={self.N})</b><br>"
            f"<span style='color:green'>Autocall: {n_auto} ({n_auto/self.N:.1%})</span><br>"
            f"<span style='color:gray'>Mature: {n_safe} ({n_safe/self.N:.1%})</span><br>"
            f"<span style='color:red'>Loss: {n_crash} ({n_crash/self.N:.1%})</span>"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.99, y=0.99,
            text=stats_text,
            showarrow=False,
            align="right",
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            borderwidth=1
        )

        fig.update_layout(
            title="Monte Carlo Path Analysis (Tunnel)",
            xaxis_title="Trading Days",
            yaxis_title="Spot Price",
            template="plotly_dark",
            showlegend=True
        )
        
        # Reset N original
        self.N = original_N
        return fig

    def plot_phoenix_distribution(self):
        """
        Histogramme de la distribution des Payoffs (actualisés).
        Montre la Value-at-Risk implicite et la moyenne (Prix).
        """
        # On utilise N simulations standard (ex: 10k ou 50k défini dans init)
        payoffs = self.calculate_payoffs_distribution()
        mean_price = np.mean(payoffs)
        
        # Conversion en % du nominal pour lecture facile
        payoffs_pct = (payoffs / self.nominal) * 100
        mean_pct = (mean_price / self.nominal) * 100
        
        fig = px.histogram(
            x=payoffs_pct, 
            nbins=60, 
            title=f"Payoff Distribution (Fair Value: {mean_pct:.2f}%)",
            color_discrete_sequence=['skyblue']
        )
        
        # Lignes verticales clés
        fig.add_vline(x=mean_pct, line_color="red", line_dash="dash", annotation_text=f"Fair Value")
        fig.add_vline(x=100, line_color="green", line_dash="dot", annotation_text="Initial Cap")

        fig.update_layout(
            xaxis_title="Payoff (% Nominal)",
            yaxis_title="Frequency",
            template="plotly_dark",
            bargap=0.1
        )
        return fig

    def plot_mc_noise_distribution(self):
        """
        Analyse de convergence : Lance 50 pricings complets pour voir la variance du prix (bruit MC).
        Sert à montrer la robustesse du prix affiché.
        """
        n_experiments = 30 # Suffisant pour la démo
        prices = []
        
        # Sauvegarde état
        original_seed = self.seed
        
        # On lance N pricings avec des seeds différents
        for i in range(n_experiments):
            self.seed = i # Change seed
            prices.append(self.price())
            
        # Reset
        self.seed = original_seed
        
        prices = np.array(prices)
        prices_pct = (prices / self.nominal) * 100
        mean = np.mean(prices_pct)
        std = np.std(prices_pct)
        
        fig = px.histogram(
            x=prices_pct,
            nbins=15,
            title=f"Monte Carlo Convergence Noise (Std Dev: {std:.2f}%)",
            color_discrete_sequence=['gray']
        )
        
        fig.add_vline(x=mean, line_color="red", line_dash="dash", annotation_text=f"Mean: {mean:.2f}%")
        
        # Zone de confiance 95%
        fig.add_vrect(
            x0=mean - 1.96*std, x1=mean + 1.96*std,
            fillcolor="yellow", opacity=0.1,
            annotation_text="95% Confidence"
        )

        fig.update_layout(
            xaxis_title="Price Estimate (% Nominal)",
            yaxis_title="Count",
            template="plotly_dark",
            bargap=0.1
        )
        return fig
    
    def plot_price_vs_strike(self, current_spot):
        """
        Trace le prix du Phoenix en fonction du niveau de Strike/Spot initial.
        Pour un produit structuré, varier le Strike revient à varier le niveau de Moneyness initial.
        """
        # On simule des variations du Spot initial de 50% à 150%
        spots = np.linspace(current_spot * 0.5, current_spot * 1.5, 50)
        prices = []
        
        # Sauvegarde du spot actuel
        original_S = self.S
        
        # Calcul pour chaque spot
        for s in spots:
            self.S = s
            # IMPORTANT: Pour le Phoenix, les barrières sont absolues.
            # Si le spot change, la "distance" aux barrières change.
            # On suppose ici que les barrières RESTENT FIXES (produit déjà émis).
            prices.append(self.price())
            
        # Restauration
        self.S = original_S
        current_price = self.price()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spots, y=prices, mode='lines', name='Price', line=dict(color='royalblue', width=2)))
        fig.add_trace(go.Scatter(x=[current_spot], y=[current_price], mode='markers', name='Current Spot', marker=dict(color='red', size=10)))
        
        fig.update_layout(
            title=" ",
            xaxis_title="Spot Price",
            yaxis_title="Phoenix Price",
            template="plotly_white",
            height=300,
            margin=dict(l=40, r=20, t=30, b=40)
        )
        return fig

    def plot_price_vs_vol(self, current_vol):
        """
        Trace le prix du Phoenix en fonction de la Volatilité.
        """
        vols = np.linspace(0.05, 0.60, 30) # De 5% à 60% de vol
        prices = []
        
        original_sigma = self.sigma
        
        for v in vols:
            self.sigma = v
            prices.append(self.price())
            
        self.sigma = original_sigma
        current_price = self.price()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vols*100, y=prices, mode='lines', name='Price', line=dict(color='orange', width=2)))
        fig.add_trace(go.Scatter(x=[current_vol*100], y=[current_price], mode='markers', name='Current Vol', marker=dict(color='red', size=10)))
        
        fig.update_layout(
            title=" ",
            xaxis_title="Volatility (%)",
            yaxis_title="Phoenix Price",
            template="plotly_white",
            height=300,
            margin=dict(l=40, r=20, t=30, b=40)
        )
        return fig
    
    def calculate_delta_quick(self, n_sims=2000):
        """
        Calcule une approximation rapide du Delta pour l'affichage temps réel.
        Utilise moins de simulations (n_sims) pour ne pas bloquer l'interface Streamlit.
        """
        # 1. Sauvegarde de l'état initial
        original_N = self.N
        original_S = self.S
        original_seed = self.seed if self.seed is not None else 42
        
        # 2. Configuration "Light"
        self.N = n_sims
        epsilon = self.S * 0.01 # Choc de 1%
        
        # 3. Calcul Prix UP
        self.S = original_S + epsilon
        self.seed = original_seed # Important: même seed pour réduire la variance
        price_up = self.price()
        
        # 4. Calcul Prix DOWN
        self.S = original_S - epsilon
        self.seed = original_seed
        price_down = self.price()
        
        # 5. Restauration de l'état
        self.S = original_S
        self.N = original_N
        self.seed = original_seed
        
        # 6. Calcul Delta (Différences finies centrées)
        delta = (price_up - price_down) / (2 * epsilon)
        
        return delta
    
    def compute_scenario_matrices(self, spot_range_pct, vol_range_abs, n_spot, n_vol, matrix_sims=1000):
        """
        Calcule les matrices de P&L pour les Heatmaps (Scénarios Spot/Vol).
        Optimisation : Utilise 'matrix_sims' (ex: 1000) au lieu de self.N pour la rapidité.
        """
        # 1. Sauvegarde État Initial
        original_N = self.N
        original_S = self.S
        original_sigma = self.sigma
        original_seed = self.seed if self.seed is not None else 42
        
        # On passe en mode "Rapide" pour la matrice
        self.N = matrix_sims
        
        # 2. Calculs de Référence (Point central 0,0)
        self.seed = original_seed
        initial_price = self.price()
        
        # Pour le hedge, on a besoin du Delta initial.
        # On utilise notre méthode rapide
        initial_delta = self.calculate_delta_quick(n_sims=matrix_sims)

        # 3. Création des Axes
        spot_moves = np.linspace(-spot_range_pct, spot_range_pct, int(n_spot))
        vol_moves = np.linspace(-vol_range_abs, vol_range_abs, int(n_vol))

        # 4. Initialisation Matrices
        matrix_unhedged = np.zeros((len(vol_moves), len(spot_moves)))
        matrix_hedged = np.zeros((len(vol_moves), len(spot_moves)))

        # 5. Boucle de Calcul (Grid Pricing)
        for i, v_chg in enumerate(vol_moves):
            for j, s_chg in enumerate(spot_moves):
                
                # Mise à jour Scénario
                self.S = original_S * (1 + s_chg)
                self.sigma = max(0.01, original_sigma + v_chg) # Sécurité vol > 1%
                self.seed = original_seed # Important: figer le seed pour comparer des pommes avec des pommes
                
                # Pricing Scénario
                new_price = self.price()

                # --- CALCUL P&L BANQUE (SHORT) ---
                # P&L = Prix Vente (Initial) - Prix Rachat (Nouveau)
                pnl_opt = initial_price - new_price
                
                # --- CALCUL HEDGE (LONG ACTIONS) ---
                # Hedge P&L = Delta * Variation Spot
                pnl_shares = initial_delta * (self.S - original_S)
                
                matrix_unhedged[i, j] = pnl_opt
                matrix_hedged[i, j] = pnl_opt + pnl_shares

        # 6. Restauration État Initial
        self.N = original_N
        self.S = original_S
        self.sigma = original_sigma
        self.seed = original_seed

        return matrix_unhedged, matrix_hedged, spot_moves, vol_moves