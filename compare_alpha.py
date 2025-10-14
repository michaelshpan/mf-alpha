#!/usr/bin/env python3
"""
Compare predicted alpha from prediction service with actual realized alpha from ETL pipeline.
This script merges the two data sources and provides comprehensive analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data(etl_path='optionA_etl/data/pilot_fact_class_month.parquet',
              pred_path='optionA_etl/predictions/alpha.parquet'):
    """Load ETL and prediction data from parquet files."""
    
    # Load ETL data with realized alpha
    print("Loading ETL data...")
    etl_data = pd.read_parquet(etl_path)
    print(f"  - Loaded {len(etl_data):,} ETL records")
    print(f"  - Date range: {etl_data['month_end'].min()} to {etl_data['month_end'].max()}")
    
    # Load prediction data
    print("\nLoading prediction data...")
    pred_data = pd.read_parquet(pred_path)
    print(f"  - Loaded {len(pred_data):,} prediction records")
    print(f"  - Date range: {pred_data['month_end'].min()} to {pred_data['month_end'].max()}")
    
    return etl_data, pred_data

def merge_and_compare(etl_data, pred_data):
    """Merge ETL and prediction data on class_id and month_end."""
    
    # Select relevant columns from ETL
    etl_subset = etl_data[['class_id', 'month_end', 'realized alpha', 'return', 
                           'total net assets', 'expense ratio', 'turnover ratio']].copy()
    
    # Rename for clarity
    etl_subset = etl_subset.rename(columns={'realized alpha': 'actual_alpha'})
    
    # Select relevant columns from predictions
    pred_subset = pred_data[['class_id', 'month_end', 'ensemble_prediction']].copy()
    pred_subset = pred_subset.rename(columns={'ensemble_prediction': 'predicted_alpha'})
    
    # Merge on class_id and month_end
    print("\nMerging datasets...")
    merged = pd.merge(pred_subset, etl_subset, 
                     on=['class_id', 'month_end'], 
                     how='inner')
    
    print(f"  - Merged {len(merged):,} matching records")
    print(f"  - {merged['class_id'].nunique()} unique funds")
    print(f"  - {merged['month_end'].nunique()} unique months")
    
    # Calculate prediction error
    merged['prediction_error'] = merged['predicted_alpha'] - merged['actual_alpha']
    merged['abs_error'] = np.abs(merged['prediction_error'])
    
    return merged

def calculate_statistics(merged_data):
    """Calculate comprehensive statistics for the comparison."""
    
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    
    # Remove rows with NaN in key columns
    clean_data = merged_data.dropna(subset=['predicted_alpha', 'actual_alpha'])
    print(f"\nAnalyzing {len(clean_data):,} records with complete alpha data")
    
    # Basic statistics
    stats_dict = {
        'Mean Actual Alpha': clean_data['actual_alpha'].mean(),
        'Mean Predicted Alpha': clean_data['predicted_alpha'].mean(),
        'Std Actual Alpha': clean_data['actual_alpha'].std(),
        'Std Predicted Alpha': clean_data['predicted_alpha'].std(),
        'Mean Prediction Error': clean_data['prediction_error'].mean(),
        'Mean Absolute Error': clean_data['abs_error'].mean(),
        'RMSE': np.sqrt((clean_data['prediction_error'] ** 2).mean()),
        'Correlation': clean_data[['predicted_alpha', 'actual_alpha']].corr().iloc[0, 1]
    }
    
    print("\nKey Metrics:")
    for key, value in stats_dict.items():
        print(f"  {key:.<30} {value:>12.6f}")
    
    # Regression analysis
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    X = clean_data[['predicted_alpha']].values
    y = clean_data['actual_alpha'].values
    
    reg = LinearRegression().fit(X, y)
    r2 = r2_score(y, reg.predict(X))
    
    print(f"\nRegression Analysis:")
    print(f"  R-squared:....................... {r2:.6f}")
    print(f"  Slope (beta):.................... {reg.coef_[0]:.6f}")
    print(f"  Intercept:....................... {reg.intercept_:.6f}")
    
    # Directional accuracy
    directional_accuracy = ((clean_data['predicted_alpha'] > 0) == 
                           (clean_data['actual_alpha'] > 0)).mean()
    print(f"\nDirectional Accuracy: {directional_accuracy:.2%}")
    
    return stats_dict, clean_data

def analyze_by_time(merged_data):
    """Analyze prediction accuracy over time."""
    
    print("\n" + "="*60)
    print("TEMPORAL ANALYSIS")
    print("="*60)
    
    monthly_stats = merged_data.groupby('month_end').agg({
        'predicted_alpha': 'mean',
        'actual_alpha': 'mean',
        'abs_error': 'mean',
        'class_id': 'count'
    }).rename(columns={'class_id': 'n_funds'})
    
    print("\nMonthly Statistics:")
    print(monthly_stats.round(6))
    
    # Calculate rolling correlation
    monthly_corr = merged_data.groupby('month_end').apply(
        lambda x: x[['predicted_alpha', 'actual_alpha']].corr().iloc[0, 1]
    )
    
    print(f"\nAverage monthly correlation: {monthly_corr.mean():.4f}")
    print(f"Correlation range: [{monthly_corr.min():.4f}, {monthly_corr.max():.4f}]")
    
    return monthly_stats

def analyze_by_fund(merged_data):
    """Analyze prediction accuracy by fund."""
    
    print("\n" + "="*60)
    print("FUND-LEVEL ANALYSIS")
    print("="*60)
    
    fund_stats = merged_data.groupby('class_id').agg({
        'predicted_alpha': ['mean', 'std'],
        'actual_alpha': ['mean', 'std'],
        'abs_error': 'mean',
        'month_end': 'count'
    }).round(6)
    
    fund_stats.columns = ['pred_mean', 'pred_std', 'actual_mean', 'actual_std', 
                          'mean_abs_error', 'n_months']
    
    # Calculate fund-level correlations
    fund_corr = merged_data.groupby('class_id').apply(
        lambda x: x[['predicted_alpha', 'actual_alpha']].corr().iloc[0, 1] 
        if len(x) > 1 else np.nan
    )
    fund_stats['correlation'] = fund_corr
    
    # Sort by prediction accuracy
    fund_stats = fund_stats.sort_values('mean_abs_error')
    
    print("\nTop 10 Best Predicted Funds:")
    print(fund_stats.head(10))
    
    print("\nTop 10 Worst Predicted Funds:")
    print(fund_stats.tail(10))
    
    print(f"\nFund-level correlation statistics:")
    print(f"  Mean: {fund_corr.mean():.4f}")
    print(f"  Median: {fund_corr.median():.4f}")
    print(f"  Std: {fund_corr.std():.4f}")
    
    return fund_stats

def create_visualizations(merged_data, output_dir='alpha_comparison'):
    """Create comprehensive visualizations of the comparison."""
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Scatter plot: Predicted vs Actual
    fig, ax = plt.subplots(figsize=(10, 8))
    clean_data = merged_data.dropna(subset=['predicted_alpha', 'actual_alpha'])
    
    # Scatter with density coloring
    hexbin = ax.hexbin(clean_data['predicted_alpha'], clean_data['actual_alpha'], 
                       gridsize=30, cmap='YlOrRd', mincnt=1)
    
    # Add perfect prediction line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k-', alpha=0.5, zorder=0, label='Perfect Prediction')
    
    # Add regression line
    z = np.polyfit(clean_data['predicted_alpha'], clean_data['actual_alpha'], 1)
    p = np.poly1d(z)
    ax.plot(clean_data['predicted_alpha'].sort_values(), 
            p(clean_data['predicted_alpha'].sort_values()), 
            "r-", alpha=0.8, label=f'Regression (slope={z[0]:.2f})')
    
    ax.set_xlabel('Predicted Alpha', fontsize=12)
    ax.set_ylabel('Actual Alpha', fontsize=12)
    ax.set_title('Predicted vs Actual Alpha Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    plt.colorbar(hexbin, ax=ax, label='Count')
    
    corr = clean_data[['predicted_alpha', 'actual_alpha']].corr().iloc[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    print(f"  - Saved scatter plot")
    
    # 2. Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of errors
    axes[0].hist(clean_data['prediction_error'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Prediction Error (Predicted - Actual)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
    
    mean_error = clean_data['prediction_error'].mean()
    std_error = clean_data['prediction_error'].std()
    axes[0].text(0.05, 0.95, 
                f'Mean: {mean_error:.6f}\nStd: {std_error:.6f}',
                transform=axes[0].transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # QQ plot
    stats.probplot(clean_data['prediction_error'], dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot of Prediction Errors', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Theoretical Quantiles', fontsize=11)
    axes[1].set_ylabel('Sample Quantiles', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  - Saved error distribution plots")
    
    # 3. Time series comparison
    monthly_avg = merged_data.groupby('month_end').agg({
        'predicted_alpha': 'mean',
        'actual_alpha': 'mean'
    }).dropna()
    
    if len(monthly_avg) > 1:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(monthly_avg.index, monthly_avg['predicted_alpha'], 
                marker='o', label='Predicted Alpha', linewidth=2)
        ax.plot(monthly_avg.index, monthly_avg['actual_alpha'], 
                marker='s', label='Actual Alpha', linewidth=2)
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Average Alpha', fontsize=12)
        ax.set_title('Monthly Average Alpha: Predicted vs Actual', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/time_series_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  - Saved time series comparison")
    
    # 4. Heatmap of correlations by month
    if merged_data['month_end'].nunique() > 3:
        monthly_corr_matrix = merged_data.pivot_table(
            index='class_id', 
            columns='month_end', 
            values='prediction_error',
            aggfunc='mean'
        )
        
        if not monthly_corr_matrix.empty and monthly_corr_matrix.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(monthly_corr_matrix, cmap='RdBu_r', center=0, 
                       cbar_kws={'label': 'Mean Prediction Error'},
                       ax=ax)
            ax.set_title('Prediction Error Heatmap by Fund and Month', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Fund ID', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/error_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"  - Saved error heatmap")
    
    plt.close('all')
    print(f"\nAll visualizations saved to {output_dir}/")

def export_results(merged_data, stats_dict, monthly_stats, fund_stats, 
                   output_file='alpha_comparison_results.xlsx'):
    """Export results to Excel for further analysis."""
    
    print("\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Raw merged data
        merged_data.to_excel(writer, sheet_name='Raw_Comparison', index=False)
        
        # Summary statistics
        pd.DataFrame([stats_dict]).T.to_excel(writer, sheet_name='Summary_Stats')
        
        # Monthly statistics
        monthly_stats.to_excel(writer, sheet_name='Monthly_Analysis')
        
        # Fund statistics
        fund_stats.to_excel(writer, sheet_name='Fund_Analysis')
        
        # Top/bottom performers
        top_funds = fund_stats.nsmallest(20, 'mean_abs_error')
        bottom_funds = fund_stats.nlargest(20, 'mean_abs_error')
        top_funds.to_excel(writer, sheet_name='Top_20_Funds')
        bottom_funds.to_excel(writer, sheet_name='Bottom_20_Funds')
    
    print(f"Results exported to {output_file}")

def main():
    """Main execution function."""
    
    print("="*60)
    print("ALPHA COMPARISON ANALYSIS")
    print("Comparing Predicted vs Actual (Realized) Alpha")
    print("="*60)
    
    # Load data
    etl_data, pred_data = load_data()
    
    # Merge and prepare comparison data
    merged_data = merge_and_compare(etl_data, pred_data)
    
    # Calculate overall statistics
    stats_dict, clean_data = calculate_statistics(merged_data)
    
    # Temporal analysis
    monthly_stats = analyze_by_time(merged_data)
    
    # Fund-level analysis
    fund_stats = analyze_by_fund(merged_data)
    
    # Create visualizations
    create_visualizations(merged_data)
    
    # Export results
    export_results(merged_data, stats_dict, monthly_stats, fund_stats)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print(f"  - Correlation between predicted and actual alpha: {stats_dict['Correlation']:.4f}")
    print(f"  - Mean Absolute Error: {stats_dict['Mean Absolute Error']:.6f}")
    print(f"  - RMSE: {stats_dict['RMSE']:.6f}")
    print(f"\nOutputs:")
    print(f"  - Excel report: alpha_comparison_results.xlsx")
    print(f"  - Visualizations: alpha_comparison/")
    
    return merged_data, stats_dict

if __name__ == "__main__":
    merged_data, stats = main()