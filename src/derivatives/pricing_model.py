import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from src.derivatives.instruments import FinancialInstrument

class EuropeanOption(FinancialInstrument):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.S = float(kwargs.get('S'))
        self.K = float(kwargs.get('K'))
        self.T = float(kwargs.get('T'))
        self.r = float(kwargs.get('r'))
        self.sigma = float(kwargs.get('sigma'))
        self.q = float(kwargs.get('q', 0.0)) 
        self.option_type = kwargs.get('option_type', 'call').lower()

    def _d1(self):
        return (np.log(self.S/self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _d2(self):
        return self._d1() - self.sigma * np.sqrt(self.T)

    def price(self):
        d1 = self._d1()
        d2 = self._d2()
        if self.option_type == "call":
            return self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
        
    def greeks(self):
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega_point(),
            "theta": self.daily_theta(),
            "rho": self.rho_point()
        }

    def delta(self):
        if self.option_type == "call":
            return np.exp(-self.q * self.T) * norm.cdf(self._d1())
        else:
            return -np.exp(-self.q * self.T) * norm.cdf(-self._d1())

    def gamma(self):
        return np.exp(-self.q * self.T) * norm.pdf(self._d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def vega_point(self):
        # Value for a 1% change in Volatility
        return (self.S * np.exp(-self.q * self.T) * norm.pdf(self._d1()) * np.sqrt(self.T)) / 100

    def daily_theta(self):
        # Value for 1 Day time decay
        d1 = self._d1()
        d2 = self._d2()
        common = -(self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        
        if self.option_type == "call":
            theta = common - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2) + self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
        else:
            theta = common + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
            
        return theta / 365

    def rho_point(self):
        # Value for 1% change in Rates
        d2 = self._d2()
        if self.option_type == "call":
            return (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)) / 100
        else:
            return -(self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)) / 100
    
    def plot_payoff(self, spot_range):
        """
        Génère un graphique interactif Plotly montrant le P&L du Client vs Banque.
        """
        spots = np.linspace(spot_range[0], spot_range[1], 100)
        premium = self.price()
        
        if self.option_type == "call":
            intrinsic_value = np.maximum(spots - self.K, 0)
        else:
            intrinsic_value = np.maximum(self.K - spots, 0)

        pnl_client = intrinsic_value - premium
        pnl_bank = premium - intrinsic_value
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=spots, y=pnl_client, 
            mode='lines', 
            name=f'Client (Long {self.option_type.title()})', 
            line=dict(color='green', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=spots, y=pnl_bank, 
            mode='lines', 
            name=f'Bank (Short {self.option_type.title()})', 
            line=dict(color='red', width=3)
        ))

        fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.5)

        fig.add_vline(
            x=self.K, 
            line_dash="dash", line_color="gray", 
            annotation_text=f"Strike ({self.K:.1f})", annotation_position="top left"
        )

        fig.add_vline(
            x=self.S, 
            line_dash="dot", line_color="cyan", 
            annotation_text=f"Current Spot ({self.S:.1f})", annotation_position="bottom right"
        )

        fig.update_layout(
            title=f" ",
            xaxis_title="Underlying price at maturity",
            yaxis_title="Profit / Loss (€)",
            template="plotly_dark", # Thème sombre pour faire ressortir le Vert/Rouge
            hovermode="x unified",   # Pour voir les deux valeurs en même temps au survol
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def plot_price_vs_strike(self, current_spot):
        """
        Affiche le prix de l'option en fonction du Strike (K).
        Ajoute un point rouge interactif indiquant la position actuelle.
        """
        # 1. Plage de Strikes (de 50% à 150% du Spot actuel)
        strikes = np.linspace(current_spot * 0.5, current_spot * 1.5, 100)
        prices = []
        
        # 2. Calcul du prix pour chaque Strike simulé
        # On garde les mêmes paramètres (vol, r, T...) sauf K qui change
        for k in strikes:
            temp_opt = EuropeanOption(
                S=self.S, K=k, T=self.T, r=self.r, sigma=self.sigma, q=self.q, option_type=self.option_type
            )
            prices.append(temp_opt.price())
            
        # 3. Récupération du point actuel (Notre Strike choisi)
        current_price = self.price()
        current_k = self.K
        
        # 4. Construction du Graphique
        fig = go.Figure()
        
        # La courbe bleue (Tous les prix possibles)
        fig.add_trace(go.Scatter(
            x=strikes, y=prices, 
            mode='lines', 
            name='Theoric Prices',
            line=dict(color='royalblue', width=2)))
        
        # Le point rouge (Notre configuration actuelle)
        fig.add_trace(go.Scatter(
            x=[current_k], y=[current_price],
            mode='markers',
            name='Your Selection',
            marker=dict(color='red', size=12, line=dict(color='white', width=2))))
        
        # Lignes guides
        fig.add_vline(x=current_spot, line_dash="dot", line_color="gray", annotation_text="Current Spot")

        fig.update_layout(
            title=" ", 
            xaxis_title="Strike Price",
            yaxis_title="Option Price (€)",
            template="plotly_white",
            height=300, # Hauteur fixe
            margin=dict(l=40, r=20, t=10, b=40), # Marges strictes
            hovermode="x unified",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        
        return fig
    
    def plot_risk_profile(self, spot_range):
        """
        Affiche les risques de couverture (Gamma & Vega) en fonction du Spot.
        C'est la vue "Hedging Difficulty".
        """
        # 1. Génération des scénarios de marché (Spot +/- 20%)
        spots = np.linspace(spot_range[0], spot_range[1], 100)
        gammas = []
        vegas = []
        
        # 2. Calcul des Grecs pour chaque scénario
        for s in spots:
            # On simule : Si le spot valait 's', quels seraient mes risques ?
            temp_opt = EuropeanOption(
                S=s, K=self.K, T=self.T, r=self.r, sigma=self.sigma, q=self.q, option_type=self.option_type
            )
            gammas.append(temp_opt.gamma())
            vegas.append(temp_opt.vega_point()) # Vega pour 1% de vol

        # 3. Récupération des valeurs actuelles pour le point rouge
        current_gamma = self.gamma()
        current_vega = self.vega_point()

        # 4. Construction du Graphique Double Axe
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # --- TRACE 1 : GAMMA (Axe Gauche - Rouge) ---
        fig.add_trace(
            go.Scatter(x=spots, y=gammas, mode='lines', name='Gamma (Convexity)', line=dict(color='crimson', width=3)),
            secondary_y=False
        )
        
        # --- TRACE 2 : VEGA (Axe Droit - Bleu pointillé) ---
        fig.add_trace(
            go.Scatter(x=spots, y=vegas, mode='lines', name='Vega (Vol Risk)', line=dict(color='royalblue', width=2, dash='dash')),
            secondary_y=True
        )

        # --- POINTS ACTUELS (Pour se situer) ---
        fig.add_trace(
            go.Scatter(x=[self.S], y=[current_gamma], mode='markers', name='Mon Gamma', marker=dict(color='crimson', size=10)),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=[self.S], y=[current_vega], mode='markers', name='Mon Vega', marker=dict(color='royalblue', size=10)),
            secondary_y=True
        )

        # 5. Mise en page
        fig.update_layout(
            title="Hedging Difficulties: Gamma & Vega Sensitivity",
            xaxis_title="Spot Price (Scenarios)",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Configuration des Axes Y
        fig.update_yaxes(title_text="Gamma", title_font=dict(color="crimson"), tickfont=dict(color="crimson"), secondary_y=False)
        fig.update_yaxes(title_text="Vega", title_font=dict(color="royalblue"), tickfont=dict(color="royalblue"), secondary_y=True)
        
        # Ligne verticale du Spot actuel
        fig.add_vline(x=self.S, line_dash="dot", line_color="gray", annotation_text="Spot Actuel")

        return fig
    
    def plot_price_vs_vol(self, current_vol):
        """Trace le prix en fonction de la Volatilité (Vega View)"""
        vols = np.linspace(0.05, 0.80, 50) # De 5% à 80% de vol
        prices = []
        
        for v in vols:
            tmp = EuropeanOption(S=self.S, K=self.K, T=self.T, r=self.r, sigma=v, q=self.q, option_type=self.option_type)
            prices.append(tmp.price())
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vols*100, y=prices, mode='lines', name='Price', line=dict(color='orange', width=3)))
        
        # Point actuel
        curr_price = self.price()
        fig.add_trace(go.Scatter(x=[current_vol*100], y=[curr_price], mode='markers', name='Current Vol', 
                                 marker=dict(color='red', size=12, line=dict(color='white', width=2))))
        
        fig.update_layout(
            title=" ", 
            xaxis_title="Volatility (%)",
            yaxis_title="Option Price (€)",
            template="plotly_white",
            height=300,
            margin=dict(l=40, r=20, t=10, b=40), # Mêmes marges = Même alignement Y
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        return fig
    
    def compute_scenario_matrices(self, spot_range_pct, vol_range_abs, n_spot, n_vol, matrix_sims=None):
        """
        Calcule les matrices de P&L.
        Note: L'argument 'matrix_sims' est présent pour la compatibilité avec l'interface
        (polymorphisme avec PhoenixStructure) mais n'est pas utilisé ici car le pricing est analytique.
        """
        # 1. État Initial
        initial_price = self.price()
        initial_delta = self.delta() 

        # 2. Création des axes (Ranges)
        spot_moves = np.linspace(-spot_range_pct, spot_range_pct, int(n_spot))
        vol_moves = np.linspace(-vol_range_abs, vol_range_abs, int(n_vol))

        # 3. Initialisation
        matrix_unhedged = np.zeros((len(vol_moves), len(spot_moves)))
        matrix_hedged = np.zeros((len(vol_moves), len(spot_moves)))

        # 4. Boucle de Calcul
        for i, v_chg in enumerate(vol_moves):
            for j, s_chg in enumerate(spot_moves):
                
                # Nouveaux paramètres
                new_S = self.S * (1 + s_chg)
                new_vol = self.sigma + v_chg
                
                if new_vol < 0.001: new_vol = 0.001

                # Pricing du nouveau scénario
                scenario_opt = EuropeanOption(
                    S=new_S, K=self.K, T=self.T, r=self.r, sigma=new_vol, q=self.q, option_type=self.option_type
                )
                new_price = scenario_opt.price()

                # P&L
                pnl_opt = -(new_price - initial_price)
                pnl_shares = initial_delta * (new_S - self.S)
                
                matrix_unhedged[i, j] = pnl_opt
                matrix_hedged[i, j] = pnl_opt + pnl_shares

        return matrix_unhedged, matrix_hedged, spot_moves, vol_moves
    

    def plot_greeks_profile(self):
        """
        Génère les graphes structurels (Delta, Gamma, Vega) avec :
        - Échelle Fixe : 0 à 200% du Strike.
        - Vue Banque : Signes inversés.
        - Indicateurs : Strike (Gris) et Spot Actuel (Rouge).
        - PAS DE THETA.
        """
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # 1. Définition de la plage fixe (0 à 200% du Strike)
        lower_bound = 0.01
        upper_bound = self.K * 2.0
        spots = np.linspace(lower_bound, upper_bound, 100)
        
        deltas, gammas, vegas = [], [], []
        
        # Sauvegarde de l'état actuel (Le point rouge)
        current_S = self.S
        
        # 2. Calculs de la courbe
        for s in spots:
            self.S = s
            # Récupération des Grecs bruts
            d = self.delta()
            g = self.gamma()
            v = self.vega_point() # ou self.vega() selon votre implémentation
            
            # Inversion VUE BANQUE (Short)
            deltas.append(-d)
            gammas.append(-g)
            vegas.append(-v)
            
        # Restauration du spot pour ne pas casser l'objet
        self.S = current_S
        
        # Calcul des valeurs exactes actuelles pour le point rouge
        curr_vals = {
            'Delta': -self.delta(),
            'Gamma': -self.gamma(),
            'Vega': -self.vega_point()
        }

        # 3. Création du Graphique (3 Lignes, 1 Colonne)
        fig = make_subplots(
            rows=3, cols=1, 
            subplot_titles=("Delta (Δ)", "Gamma (Γ)", "Vega (ν)"),
            shared_xaxes=True, # Axe X partagé pour mieux comparer
            vertical_spacing=0.05
        )

        # Helper pour tracer proprement
        def add_trace_with_markers(row, col, x_data, y_data, name, current_val):
            # A. La Courbe Bleue
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=name, 
                                     line=dict(color='#1f77b4', width=2), showlegend=False), 
                          row=row, col=col)
            
            # B. Le Strike (Ligne pointillée Grise)
            fig.add_vline(x=self.K, line_width=1, line_dash="dash", line_color="gray", row=row, col=col)
            
            # Annotation Strike (uniquement sur le dernier graph pour clarté)
            if row == 3: 
                 fig.add_annotation(x=self.K, y=min(y_data), text="Strike", showarrow=False, yshift=-10, font=dict(size=10, color="gray"), row=row, col=col)

            # C. Le Spot Actuel (Point Rouge)
            fig.add_trace(go.Scatter(
                x=[current_S], y=[current_val], mode='markers', 
                marker=dict(color='red', size=8, symbol='circle'),
                name="Current", showlegend=False
            ), row=row, col=col)

        # Ajout des 3 traces (Delta, Gamma, Vega)
        add_trace_with_markers(1, 1, spots, deltas, "Delta", curr_vals['Delta'])
        add_trace_with_markers(2, 1, spots, gammas, "Gamma", curr_vals['Gamma'])
        add_trace_with_markers(3, 1, spots, vegas, "Vega", curr_vals['Vega'])

        # Mise en page finale
        fig.update_layout(height=700, title_text="Greeks Structural Profile (Bank View)", margin=dict(t=60, b=20, l=20, r=20))
        fig.update_xaxes(title_text="Spot Price", range=[0, upper_bound], row=3, col=1)
        
        return fig