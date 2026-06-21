#!/usr/bin/env python3
"""
Generate figures from saved simulation results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from src.visualization import FigureGenerator

def main():
    """
    Load saved results and generate figures.
    """
    # Load results
    with open('data/results_debt.pkl', 'rb') as f:
        results_debt = pickle.load(f)
    
    with open('data/results_yusuf.pkl', 'rb') as f:
        results_yusuf = pickle.load(f)
    
    with open('data/mc_results.pkl', 'rb') as f:
        mc_results = pickle.load(f)
    
    # Generate figures
    fig_gen = FigureGenerator()
    fig_gen.figure_all(results_debt, results_yusuf, mc_results)
    
    print("Figures generated successfully!")

if __name__ == "__main__":
    main()
