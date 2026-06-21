"""
Model equations for the thermodynamic bifurcation parameter Lambda.
Contains the core mathematical model for debt-based monetary systems.
"""

import numpy as np
from scipy.integrate import odeint

class ThermodynamicModel:
    """
    Core model for the thermodynamic bifurcation parameter Lambda.
    
    State variables:
        D(t): Total debt (monetary units)
        S(t): Stock of essential goods (units)
        P(t): Production rate (units/year)
        C(t): Consumption rate (units/year)
    
    Parameters:
        D0: Initial debt (monetary units)
        S0: Initial stock (units)
        P0: Initial production (units/year)
        r: Interest rate (1/year)
        alpha: Production growth rate (1/year)
        nu: Basic consumption needs (units/year)
        phi: Fraction of dissipated interest (0 <= phi <= 1)
        E_max: Maximum low-entropy extraction (J/year)
        p_E: Energy price (monetary units / J)
        kappa: Fraction of P-C covered by E_low (0 < kappa <= 1)
    """
    
    def __init__(self, D0=100.0, S0=10.0, P0=1.0,
                 r=0.05, alpha=0.02, nu=0.5,
                 phi=0.8, E_max=10.0, p_E=1.0, kappa=0.5):
        
        self.D0 = D0
        self.S0 = S0
        self.P0 = P0
        self.r = r
        self.alpha = alpha
        self.nu = nu
        self.phi = phi
        self.E_max = E_max
        self.p_E = p_E
        self.kappa = kappa
        
    def debt_dynamics(self, D, t):
        """Debt dynamics: dD/dt = r * D"""
        return self.r * D
    
    def debt_solution(self, t):
        """Exact solution for debt: D(t) = D0 * exp(r * t)"""
        return self.D0 * np.exp(self.r * t)
    
    def production_solution(self, t):
        """Exact solution for production: P(t) = P0 * exp(alpha * t)"""
        return self.P0 * np.exp(self.alpha * t)
    
    def consumption(self, P, D, t):
        """
        Consumption function: C(t) = min(P(t) - phi*r*D(t), nu)
        """
        debt_service = self.phi * self.r * D
        available = P - debt_service
        return np.minimum(available, self.nu)
    
    def lambda_parameter(self, D, t):
        """
        Bifurcation parameter: Lambda = (D * r) / (p_E * E_low)
        """
        return (D * self.r) / (self.p_E * self.E_max)
    
    def stock_dynamics(self, state, t):
        """
        Stock dynamics: dS/dt = P - C - phi*r*D
        """
        D, S = state
        
        # Production
        P = self.production_solution(t)
        
        # Consumption
        C = self.consumption(P, D, t)
        
        # Stock derivative
        dDdt = self.debt_dynamics(D, t)
        dSdt = P - C - self.phi * self.r * D
        
        return [dDdt, dSdt]
    
    def simulate(self, T=100.0, dt=0.01):
        """
        Run full simulation.
        
        Args:
            T: Total simulation time (years)
            dt: Time step (years)
        
        Returns:
            dict containing time series for all variables
        """
        # Time vector
        t = np.arange(0, T + dt, dt)
        n_steps = len(t)
        
        # Initialize arrays
        D = np.zeros(n_steps)
        S = np.zeros(n_steps)
        P = np.zeros(n_steps)
        C = np.zeros(n_steps)
        Lambda = np.zeros(n_steps)
        
        # Initial conditions
        D[0] = self.D0
        S[0] = self.S0
        
        # Integration
        for i in range(n_steps - 1):
            # Production
            P[i] = self.production_solution(t[i])
            
            # Consumption
            C[i] = self.consumption(P[i], D[i], t[i])
            
            # Debt dynamics (exact)
            D[i+1] = self.debt_solution(t[i+1])
            
            # Lambda
            Lambda[i] = self.lambda_parameter(D[i], t[i])
            
            # Stock dynamics (Euler integration)
            dSdt = P[i] - C[i] - self.phi * self.r * D[i]
            S[i+1] = S[i] + dSdt * dt
            
            # Prevent negative stock
            if S[i+1] < 0:
                S[i+1] = 0
                # Stop simulation if stock collapses
                break
        
        # Fill remaining arrays
        P[n_steps-1] = self.production_solution(t[n_steps-1])
        C[n_steps-1] = self.consumption(P[n_steps-1], D[n_steps-1], t[n_steps-1])
        Lambda[n_steps-1] = self.lambda_parameter(D[n_steps-1], t[n_steps-1])
        
        return {
            't': t,
            'D': D,
            'S': S,
            'P': P,
            'C': C,
            'Lambda': Lambda,
            'T_cross': self.find_crossing_time(Lambda, t)
        }
    
    def find_crossing_time(self, Lambda, t):
        """Find time when Lambda crosses unity"""
        idx = np.where(Lambda > 1.0)[0]
        if len(idx) > 0:
            return t[idx[0]]
        return np.inf
    
    def compute_metrics(self, results):
        """
        Compute key metrics from simulation results.
        """
        S = results['S']
        Lambda = results['Lambda']
        t = results['t']
        
        # Final stock
        S_final = S[-1]
        
        # Time of Lambda crossing
        T_cross = self.find_crossing_time(Lambda, t)
        
        # Erosion rate (exponential fit after crossing)
        idx_cross = np.where(t >= T_cross)[0]
        if len(idx_cross) > 1:
            S_erosion = S[idx_cross]
            t_erosion = t[idx_cross]
            # Fit: S(t) = S0 * exp(-lambda_erosion * t)
            log_S = np.log(S_erosion + 1e-10)
            coeff = np.polyfit(t_erosion, log_S, 1)
            lambda_erosion = -coeff[0]
        else:
            lambda_erosion = np.nan
        
        return {
            'S_final': S_final,
            'T_cross': T_cross,
            'lambda_erosion': lambda_erosion
        }
