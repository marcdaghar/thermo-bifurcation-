#!/usr/bin/env python3
"""
Main simulation script for the thermodynamic bifurcation model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from src.model import ThermodynamicModel
from src.simulation import MonteCarloSimulator
from src.visualization import FigureGenerator

def run_single_simulation():
    """
    Run a single simulation of the debt-based system.
    """
    print("=" * 60)
    print("Running single simulation...")
    print("=" * 60)
    
    # Initialize model
    model = ThermodynamicModel(
        D0=100.0,
        S0=10.0,
        P0=1.0,
        r=0.05,
        alpha=0.02,
        nu=0.5,
        phi=0.8,
        E_max=10.0,
        p_E=1.0,
        kappa=0.5
    )
    
    # Run simulation
    results_debt = model.simulate(T=100.0, dt=0.01)
    
    # Compute metrics
    metrics = model.compute_metrics(results_debt)
    
    print(f"\nDebt-based system results:")
    print(f"  Crossing time T_c = {metrics['T_cross']:.2f} years")
    print(f"  Final stock S(T) = {metrics['S_final']:.4f}")
    print(f"  Erosion rate lambda = {metrics['lambda_erosion']:.4f} 1/year")
    
    return results_debt

def run_yusuf_simulation():
    """
    Run a simulation of the Yusuf-Grondona (zero-debt) system.
    """
    print("\n" + "=" * 60)
    print("Running Yusuf-Grondona simulation (zero debt)...")
    print("=" * 60)
    
    # For Yusuf-Grondona, set D0 = 0
    model = ThermodynamicModel(
        D0=0.0,
        S0=10.0,
        P0=1.0,
        r=0.05,
        alpha=0.02,
        nu=0.5,
        phi=0.8,
        E_max=10.0,
        p_E=1.0,
        kappa=0.5
    )
    
    # Run simulation
    results_yusuf = model.simulate(T=100.0, dt=0.01)
    
    print(f"\nYusuf-Grondona system results:")
    print(f"  Final stock S(T) = {results_yusuf['S'][-1]:.4f}")
    print(f"  Lambda(t) = {results_yusuf['Lambda'][-1]:.4f}")
    
    return results_yusuf

def run_monte_carlo():
    """
    Run Monte Carlo sensitivity analysis.
    """
    print("\n" + "=" * 60)
    print("Running Monte Carlo sensitivity analysis...")
    print("=" * 60)
    
    # Define parameter ranges
    param_ranges = {
        'r': (0.02, 0.08),      # Interest rate: 2-8%
        'E_max': (5.0, 15.0),   # Max extraction: 5-15 J/year
        'phi': (0.6, 1.0),      # Dissipation fraction: 60-100%
        'kappa': (0.3, 0.7)     # P-C fraction: 30-70%
    }
    
    # Initialize Monte Carlo
    mc = MonteCarloSimulator(n_runs=500)
    
    # Run simulations
    mc_results = mc.run_simulation_batch(param_ranges, T=100.0, dt=0.01)
    
    # Analyze results
    analysis = mc.analyze_results(mc_results)
    
    print(f"\nMonte Carlo analysis results:")
    print(f"  T_cross: mean = {analysis['T_cross']['mean']:.2f}, "
          f"std = {analysis['T_cross']['std']:.2f}")
    print(f"  T_cross: median = {analysis['T_cross']['median']:.2f}, "
          f"95% CI: [{analysis['T_cross']['q25']:.2f}, {analysis['T_cross']['q75']:.2f}]")
    print(f"  S_final: mean = {analysis['S_final']['mean']:.4f}, "
          f"std = {analysis['S_final']['std']:.4f}")
    
    return mc_results

def main():
    """
    Main execution function.
    """
    # Run simulations
    results_debt = run_single_simulation()
    results_yusuf = run_yusuf_simulation()
    mc_results = run_monte_carlo()
    
    # Save results
    os.makedirs('data', exist_ok=True)
    with open('data/results_debt.pkl', 'wb') as f:
        pickle.dump(results_debt, f)
    with open('data/results_yusuf.pkl', 'wb') as f:
        pickle.dump(results_yusuf, f)
    with open('data/mc_results.pkl', 'wb') as f:
        pickle.dump(mc_results, f)
    
    print("\n" + "=" * 60)
    print("Results saved to data/ directory.")
    print("=" * 60)
    
    # Generate figures
    print("\nGenerating figures...")
    fig_gen = FigureGenerator()
    fig_gen.figure_all(results_debt, results_yusuf, mc_results)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
