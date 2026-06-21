"""
Monte Carlo simulation framework for sensitivity analysis.
"""

import numpy as np
from tqdm import tqdm
from src.model import ThermodynamicModel

class MonteCarloSimulator:
    """
    Monte Carlo simulation for parameter sensitivity analysis.
    """
    
    def __init__(self, base_params=None, n_runs=1000):
        """
        Args:
            base_params: Dictionary of base parameters
            n_runs: Number of Monte Carlo runs
        """
        if base_params is None:
            base_params = {
                'D0': 100.0,
                'S0': 10.0,
                'P0': 1.0,
                'r': 0.05,
                'alpha': 0.02,
                'nu': 0.5,
                'phi': 0.8,
                'E_max': 10.0,
                'p_E': 1.0,
                'kappa': 0.5
            }
        self.base_params = base_params
        self.n_runs = n_runs
        
    def generate_parameter_samples(self, param_ranges):
        """
        Generate random parameter samples within given ranges.
        
        Args:
            param_ranges: Dictionary of (min, max) for each parameter
        
        Returns:
            List of parameter dictionaries
        """
        samples = []
        for _ in range(self.n_runs):
            params = self.base_params.copy()
            for key, (min_val, max_val) in param_ranges.items():
                params[key] = np.random.uniform(min_val, max_val)
            samples.append(params)
        return samples
    
    def run_simulation_batch(self, param_ranges, T=100.0, dt=0.01):
        """
        Run batch of simulations with parameter variations.
        
        Args:
            param_ranges: Dictionary of (min, max) for each parameter
            T: Simulation time
            dt: Time step
        
        Returns:
            List of results dictionaries
        """
        param_samples = self.generate_parameter_samples(param_ranges)
        results = []
        
        for params in tqdm(param_samples, desc="Running simulations"):
            model = ThermodynamicModel(**params)
            sim_results = model.simulate(T=T, dt=dt)
            metrics = model.compute_metrics(sim_results)
            results.append({
                'params': params,
                'sim': sim_results,
                'metrics': metrics
            })
        
        return results
    
    def analyze_results(self, results):
        """
        Analyze Monte Carlo results.
        
        Returns:
            Dictionary with summary statistics
        """
        T_cross_list = []
        S_final_list = []
        lambda_erosion_list = []
        
        for r in results:
            T_cross_list.append(r['metrics']['T_cross'])
            S_final_list.append(r['metrics']['S_final'])
            lambda_erosion_list.append(r['metrics']['lambda_erosion'])
        
        # Filter out infinities
        T_cross_finite = [x for x in T_cross_list if np.isfinite(x)]
        
        return {
            'T_cross': {
                'mean': np.mean(T_cross_finite),
                'std': np.std(T_cross_finite),
                'median': np.median(T_cross_finite),
                'q25': np.percentile(T_cross_finite, 25),
                'q75': np.percentile(T_cross_finite, 75)
            },
            'S_final': {
                'mean': np.mean(S_final_list),
                'std': np.std(S_final_list)
            },
            'lambda_erosion': {
                'mean': np.mean([x for x in lambda_erosion_list if not np.isnan(x)]),
                'std': np.std([x for x in lambda_erosion_list if not np.isnan(x)])
            }
        }
