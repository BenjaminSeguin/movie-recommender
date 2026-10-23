import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'


def plot_option_1_modern_clean(results, ratios, k_bounded, k_userreg, out_path):
    """
    Option 1: Modern, clean professional style
    - Sophisticated color palette
    - Better typography and spacing
    - Gradient background
    - Enhanced visual hierarchy
    """
    # Modern color palette (colorblind-friendly)
    colors = {
        'Bounded ALS': '#3498db',    # Bright blue
        'UserReg': '#e74c3c',        # Vibrant red
        'HERec': '#2ecc71'          # Fresh green
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Subtle gradient background
    ax.set_facecolor('#fafafa')
    
    x = np.array([int(100 * r) for r in ratios])
    
    # Plot each model with enhanced styling
    for model_name in results.keys():
        means = np.array(results[model_name]['means'])
        stds = np.array(results[model_name]['stds'])
        color = colors[model_name]
        
        # Main line with shadow effect
        ax.plot(x, means, 
               color=color, linewidth=3, 
               marker='o', markersize=12, markeredgewidth=2, markeredgecolor='white',
               label=model_name, zorder=3, alpha=0.9)
        
        # Confidence interval with gradient
        ax.fill_between(x, means - stds, means + stds,
                       color=color, alpha=0.15, linewidth=0, zorder=1)
        
        # Add a thin border to confidence interval
        ax.plot(x, means - stds, color=color, linewidth=0.5, alpha=0.3, linestyle='--')
        ax.plot(x, means + stds, color=color, linewidth=0.5, alpha=0.3, linestyle='--')
    
    # Styling
    ax.set_xlabel('Training Data Ratio (%)', fontsize=16, fontweight='600', labelpad=10)
    ax.set_ylabel('Test RMSE', fontsize=16, fontweight='600', labelpad=10)
    ax.set_title('Cross-Validation Performance Comparison', 
                fontsize=20, fontweight='700', pad=20, loc='left')
    
    # Subtitle with configuration
    subtitle = f'Bounded ALS (k={k_bounded}) • UserReg (k={k_userreg}) • HERec (HIN-based)'
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, 
           fontsize=11, style='italic', color='#666')
    
    # Enhanced grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.8, color='#cccccc')
    ax.set_axisbelow(True)
    
    # Legend with custom styling
    legend = ax.legend(loc='upper right', fontsize=13, framealpha=0.95, 
                      edgecolor='#ddd', fancybox=True, shadow=True)
    legend.get_frame().set_linewidth(1.5)
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.5)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#ccc')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Modern clean plot saved to {out_path}")


def plot_option_2_panel_view(results, ratios, k_bounded, k_userreg, out_path):
    """
    Option 2: Multi-panel view with insights
    - Main comparison plot
    - Performance improvement subplot
    - Final performance bar chart
    """
    colors = {
        'Bounded ALS': '#FF6B6B',
        'UserReg': '#4ECDC4', 
        'HERec': '#95E1D3'
    }
    
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 1], hspace=0.35, wspace=0.3)
    
    # Main plot (spans 2 rows, 2 columns)
    ax_main = fig.add_subplot(gs[:, :2])
    
    x = np.array([int(100 * r) for r in ratios])
    
    for model_name in results.keys():
        means = np.array(results[model_name]['means'])
        stds = np.array(results[model_name]['stds'])
        color = colors[model_name]
        
        ax_main.plot(x, means, color=color, linewidth=3, 
                    marker='o', markersize=10, label=model_name, markeredgewidth=2,
                    markeredgecolor='white', zorder=3)
        ax_main.fill_between(x, means - stds, means + stds,
                            color=color, alpha=0.2, linewidth=0)
    
    ax_main.set_xlabel('Training Ratio (%)', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Test RMSE', fontsize=14, fontweight='bold')
    ax_main.set_title('Model Performance vs. Training Data', 
                     fontsize=16, fontweight='bold', pad=15)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.legend(fontsize=11, loc='best', frameon=True, shadow=True)
    
    # Improvement subplot (top right)
    ax_improve = fig.add_subplot(gs[0, 2])
    
    # Calculate improvement from 20% to 80% training
    improvements = {}
    for model_name in results.keys():
        means = np.array(results[model_name]['means'])
        improvement = ((means[0] - means[-1]) / means[0]) * 100
        improvements[model_name] = improvement
    
    bars = ax_improve.bar(range(len(improvements)), list(improvements.values()),
                          color=[colors[k] for k in improvements.keys()],
                          alpha=0.8, edgecolor='white', linewidth=2)
    
    ax_improve.set_xticks(range(len(improvements)))
    ax_improve.set_xticklabels([name.replace(' ', '\n') for name in improvements.keys()], 
                               fontsize=9, rotation=0)
    ax_improve.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax_improve.set_title('RMSE Reduction\n(20%→80% data)', fontsize=11, fontweight='bold')
    ax_improve.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, val in zip(bars, improvements.values()):
        height = bar.get_height()
        ax_improve.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}%', ha='center', va='bottom', 
                       fontweight='bold', fontsize=10)
    
    # Best performance comparison (bottom right)
    ax_best = fig.add_subplot(gs[1, 2])
    
    best_rmse = {name: min(results[name]['means']) for name in results.keys()}
    bars = ax_best.barh(range(len(best_rmse)), list(best_rmse.values()),
                       color=[colors[k] for k in best_rmse.keys()],
                       alpha=0.8, edgecolor='white', linewidth=2)
    
    ax_best.set_yticks(range(len(best_rmse)))
    ax_best.set_yticklabels(list(best_rmse.keys()), fontsize=10)
    ax_best.set_xlabel('Best RMSE', fontsize=11, fontweight='bold')
    ax_best.set_title('Optimal Performance', fontsize=11, fontweight='bold')
    ax_best.grid(axis='x', alpha=0.3, linestyle='--')
    ax_best.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, best_rmse.values()):
        width = bar.get_width()
        ax_best.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{val:.4f}', ha='left', va='center', 
                    fontweight='bold', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Panel view plot saved to {out_path}")


def plot_option_3_focused_comparison(results, ratios, k_bounded, k_userreg, out_path):
    """
    Option 3: Focused comparison with annotations
    - Highlight best model
    - Annotate key insights
    - Show crossover points
    """
    colors = ['#6C5CE7', '#FD79A8', '#00B894']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Background styling
    ax.set_facecolor('#f8f9fa')
    
    x = np.array([int(100 * r) for r in ratios])
    model_names = list(results.keys())
    
    # Find best model at 80%
    best_at_80 = min(model_names, key=lambda m: results[m]['means'][-1])
    
    for i, model_name in enumerate(model_names):
        means = np.array(results[model_name]['means'])
        stds = np.array(results[model_name]['stds'])
        color = colors[i]
        
        # Highlight best model
        if model_name == best_at_80:
            linewidth = 4
            alpha = 1.0
            zorder = 5
            marker_size = 14
        else:
            linewidth = 2.5
            alpha = 0.7
            zorder = 3
            marker_size = 10
        
        ax.plot(x, means, color=color, linewidth=linewidth, alpha=alpha,
               marker='o', markersize=marker_size, label=model_name, zorder=zorder,
               markeredgewidth=2, markeredgecolor='white')
        ax.fill_between(x, means - stds, means + stds,
                       color=color, alpha=0.15, linewidth=0)
        
        # Annotate final performance
        final_rmse = means[-1]
        ax.annotate(f'{final_rmse:.3f}', 
                   xy=(x[-1], final_rmse), 
                   xytext=(10, 0), textcoords='offset points',
                   fontsize=11, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                           edgecolor=color, alpha=0.9, linewidth=2))
    
    # Add champion badge
    best_means = results[best_at_80]['means']
    champion_x = x[-1]
    champion_y = best_means[-1]
    
    # Star marker for best
    ax.plot(champion_x, champion_y, marker='*', markersize=25, 
           color='gold', markeredgecolor='#ff6b6b', markeredgewidth=2, zorder=10)
    
    # Title with insight
    improvement = ((best_means[0] - best_means[-1]) / best_means[0]) * 100
    title = f'Model Performance Comparison'
    subtitle = f'{best_at_80} achieves best performance with {improvement:.1f}% RMSE reduction'
    
    ax.text(0.5, 1.08, title, transform=ax.transAxes, 
           fontsize=22, fontweight='bold', ha='center')
    ax.text(0.5, 1.03, subtitle, transform=ax.transAxes, 
           fontsize=12, style='italic', ha='center', color='#555')
    
    ax.set_xlabel('Training Data Ratio (%)', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Test RMSE', fontsize=15, fontweight='bold', labelpad=10)
    
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax.legend(fontsize=13, loc='upper right', framealpha=0.95, 
             edgecolor='#ccc', shadow=True, fancybox=True)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Focused comparison plot saved to {out_path}")


def plot_option_4_minimalist(results, ratios, k_bounded, k_userreg, out_path):
    """
    Option 4: Minimalist publication style
    - Clean, simple design
    - High contrast
    - Publication-ready
    """
    # Use seaborn style as base
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Minimalist color palette (grayscale + accent)
    colors = {
        'Bounded ALS': '#2d3436',
        'UserReg': '#636e72',
        'HERec': '#e17055'  # Accent color for best/new method
    }
    
    x = np.array([int(100 * r) for r in ratios])
    
    for model_name in results.keys():
        means = np.array(results[model_name]['means'])
        stds = np.array(results[model_name]['stds'])
        color = colors[model_name]
        
        # Simple clean lines
        ax.plot(x, means, color=color, linewidth=2.5, 
               marker='o', markersize=8, label=model_name,
               markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
        
        # Subtle confidence bands
        ax.fill_between(x, means - stds, means + stds,
                       color=color, alpha=0.1, linewidth=0)
    
    ax.set_xlabel('Training Ratio (%)', fontsize=13)
    ax.set_ylabel('Test RMSE', fontsize=13)
    ax.set_title('Stratified Per-User Cross-Validation', 
                fontsize=15, fontweight='600', pad=15)
    
    # Minimal legend
    ax.legend(fontsize=11, frameon=True, loc='best', 
             framealpha=1.0, edgecolor='#ddd')
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Minimalist plot saved to {out_path}")

import os
# ============= INTEGRATION FUNCTION =============

def create_all_visualizations(results, ratios, k_bounded, k_userreg, base_path='results/cv'):
    """Create all visualization options"""
    os.makedirs(os.path.dirname(base_path) if os.path.dirname(base_path) else 'results', exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating visualization options...")
    print("="*60)
    
    plot_option_1_modern_clean(results, ratios, k_bounded, k_userreg, 
                               f'{base_path}_option1_modern.png')
    
    plot_option_2_panel_view(results, ratios, k_bounded, k_userreg, 
                            f'{base_path}_option2_panel.png')
    
    plot_option_3_focused_comparison(results, ratios, k_bounded, k_userreg, 
                                    f'{base_path}_option3_focused.png')
    
    plot_option_4_minimalist(results, ratios, k_bounded, k_userreg, 
                            f'{base_path}_option4_minimalist.png')
    
    print("\n✨ All visualization options created successfully!")
    print(f"   Check the results folder for 4 different styles")