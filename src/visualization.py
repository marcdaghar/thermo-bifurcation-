"""
Visualization functions for generating publication-ready figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300

class FigureGenerator:
    """
    Generate figures for the thermodynamic bifurcation article.
    """
    
    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def figure_lambda_trajectory(self, results):
        """
        Figure 1a: Evolution of Lambda(t)
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        t = results['t']
        Lambda = results['Lambda']
        
        # Plot Lambda
        ax.plot(t, Lambda, linewidth=2.5, color='darkred', label=r'$\Lambda(t)$')
        
        # Horizontal line at Lambda = 1
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, 
                   label=r'$\Lambda = 1$ (critical threshold)')
        
        # Crossing time
        T_cross = results['T_cross']
        if np.isfinite(T_cross):
            ax.axvline(x=T_cross, color='gray', linestyle=':', linewidth=1.5,
                       label=f'$T_c = {T_cross:.1f}$ years')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel(r'Bifurcation parameter $\Lambda(t)$', fontsize=12)
        ax.set_title('Evolution of the Bifurcation Parameter', fontsize=14)
        ax.legend(loc='upper left')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/lambda_trajectory.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/lambda_trajectory.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_stock_trajectory(self, results):
        """
        Figure 1b: Evolution of S(t)
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        t = results['t']
        S = results['S']
        Lambda = results['Lambda']
        T_cross = results['T_cross']
        
        # Plot S(t)
        ax.plot(t, S, linewidth=2.5, color='navy', label=r'$S(t)$ (stock)')
        
        # Mark crossing time
        if np.isfinite(T_cross):
            ax.axvline(x=T_cross, color='gray', linestyle=':', linewidth=1.5,
                       label=f'$\\Lambda = 1$ at $t = {T_cross:.1f}$ years')
            
            # Shade erosion regime
            ax.axvspan(T_cross, t[-1], alpha=0.2, color='red', 
                       label='Erosion regime')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel(r'Essential stock $S(t)$', fontsize=12)
        ax.set_title('Evolution of the Essential Stock', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/stock_trajectory.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/stock_trajectory.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_comparison(self, results_debt, results_yusuf):
        """
        Figure 2: Comparison of debt-based vs. Yusuf-Grondona systems
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Debt-based system
        t1 = results_debt['t']
        S1 = results_debt['S']
        ax.plot(t1, S1, linewidth=2.5, color='red', 
                label='Debt-based system (erosion)')
        
        # Yusuf-Grondona system
        t2 = results_yusuf['t']
        S2 = results_yusuf['S']
        ax.plot(t2, S2, linewidth=2.5, color='blue', 
                label='Yusuf-Grondona system (stable)')
        
        # Mark stable level
        S_stable = S2[-1]
        ax.axhline(y=S_stable, color='blue', linestyle='--', linewidth=1.5,
                   label=f'$S^* = {S_stable:.2f}$ (stable equilibrium)')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel(r'Essential stock $S(t)$', fontsize=12)
        ax.set_title('Comparative Stock Dynamics', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comparison.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/comparison.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_lambda_vs_stock(self, results):
        """
        Phase portrait: Lambda vs S
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        Lambda = results['Lambda']
        S = results['S']
        
        ax.plot(Lambda, S, linewidth=2, color='purple', alpha=0.8)
        
        # Mark crossing
        T_cross = results['T_cross']
        if np.isfinite(T_cross):
            idx_cross = int(T_cross / 0.01)
            ax.plot(Lambda[idx_cross], S[idx_cross], 'o', color='red', 
                    markersize=8, label=f'$\\Lambda = 1$ at $t = {T_cross:.1f}$')
        
        ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5)
        ax.set_xlabel(r'Bifurcation parameter $\Lambda$', fontsize=12)
        ax.set_ylabel(r'Essential stock $S$', fontsize=12)
        ax.set_title('Phase Portrait: $\Lambda$ vs $S$', fontsize=14)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/lambda_vs_stock.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/lambda_vs_stock.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_monte_carlo(self, mc_results):
        """
        Monte Carlo sensitivity analysis results.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        T_cross = []
        S_final = []
        lambda_erosion = []
        r_values = []
        E_values = []
        
        for r in mc_results:
            T_cross.append(r['metrics']['T_cross'])
            S_final.append(r['metrics']['S_final'])
            lambda_erosion.append(r['metrics']['lambda_erosion'])
            r_values.append(r['params']['r'])
            E_values.append(r['params']['E_max'])
        
        # Filter finite values for T_cross
        valid = np.isfinite(T_cross)
        T_cross_valid = np.array(T_cross)[valid]
        r_valid = np.array(r_values)[valid]
        E_valid = np.array(E_values)[valid]
        
        # Panel 1: T_cross vs r
        ax = axes[0, 0]
        ax.scatter(r_valid, T_cross_valid, alpha=0.5, s=10, color='darkblue')
        ax.set_xlabel('Interest rate $r$ (1/year)')
        ax.set_ylabel('Crossing time $T_c$ (years)')
        ax.set_title('Sensitivity to interest rate')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: T_cross vs E_max
        ax = axes[0, 1]
        ax.scatter(E_valid, T_cross_valid, alpha=0.5, s=10, color='darkgreen')
        ax.set_xlabel('Max extraction $E_{max}$ (J/year)')
        ax.set_ylabel('Crossing time $T_c$ (years)')
        ax.set_title('Sensitivity to resource limit')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Distribution of T_cross
        ax = axes[1, 0]
        ax.hist(T_cross_valid, bins=30, color='darkorange', alpha=0.7, edgecolor='black')
        ax.axvline(np.median(T_cross_valid), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(T_cross_valid):.1f}')
        ax.set_xlabel('Crossing time $T_c$ (years)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of crossing times')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 4: S_final distribution
        ax = axes[1, 1]
        ax.hist(S_final, bins=30, color='darkred', alpha=0.7, edgecolor='black')
        ax.axvline(np.median(S_final), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(S_final):.2f}')
        ax.set_xlabel('Final stock $S(T)$')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of final stock')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/monte_carlo_results.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/monte_carlo_results.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_all(self, results_debt, results_yusuf, mc_results):
        """
        Generate all figures for the article.
        """
        print("Generating Figure 1a: Lambda trajectory...")
        self.figure_lambda_trajectory(results_debt)
        
        print("Generating Figure 1b: Stock trajectory...")
        self.figure_stock_trajectory(results_debt)
        
        print("Generating Figure 2: Comparison...")
        self.figure_comparison(results_debt, results_yusuf)
        
        print("Generating Figure 3: Lambda vs Stock...")
        self.figure_lambda_vs_stock(results_debt)
        
        print("Generating Figure 4: Monte Carlo results...")
        self.figure_monte_carlo(mc_results)
        
        print("All figures generated successfully!")
