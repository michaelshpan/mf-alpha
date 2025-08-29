#!/usr/bin/env python3
"""
Compare prediction results across funds and time
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_predictions(prediction_file: str = "predictions/alpha.parquet"):
    """Comprehensive analysis of prediction results"""
    
    print("🔍 PREDICTION ANALYSIS REPORT")
    print("=" * 50)
    
    # Load data
    df = pd.read_parquet(prediction_file)
    
    print(f"📊 Dataset: {len(df)} predictions across {df['class_id'].nunique()} funds")
    print(f"📅 Period: {df['month_end'].min()} to {df['month_end'].max()}")
    
    # Fund-level analysis
    print(f"\n🏢 FUND-LEVEL ANALYSIS:")
    print("-" * 30)
    
    fund_stats = df.groupby('class_id')['ensemble_prediction'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    fund_stats.columns = ['Observations', 'Mean Alpha', 'Std Dev', 'Min Alpha', 'Max Alpha']
    print(fund_stats)
    
    # Time-based analysis
    if 'month_end' in df.columns:
        print(f"\n📈 TIME-BASED ANALYSIS:")
        print("-" * 30)
        
        df['month_end'] = pd.to_datetime(df['month_end'])
        monthly_stats = df.groupby('month_end')['ensemble_prediction'].agg([
            'count', 'mean', 'std'
        ]).round(4)
        monthly_stats.columns = ['Funds', 'Mean Alpha', 'Std Dev']
        print(monthly_stats.head(10))
        
        if len(monthly_stats) > 10:
            print(f"... and {len(monthly_stats) - 10} more months")
    
    # Distribution analysis
    print(f"\n📊 DISTRIBUTION ANALYSIS:")
    print("-" * 30)
    
    predictions = df['ensemble_prediction']
    print(f"Mean: {predictions.mean():.4f}")
    print(f"Median: {predictions.median():.4f}")
    print(f"Std Dev: {predictions.std():.4f}")
    print(f"Skewness: {predictions.skew():.4f}")
    print(f"Kurtosis: {predictions.kurtosis():.4f}")
    
    # Percentiles
    percentiles = predictions.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"\nPercentiles:")
    for p, val in percentiles.items():
        print(f"  {p*100:4.0f}%: {val:.4f}")
    
    # Top and bottom performers
    print(f"\n🏆 TOP PERFORMERS (by mean alpha):")
    top_funds = fund_stats.nlargest(3, 'Mean Alpha')
    for fund_id, stats in top_funds.iterrows():
        print(f"  {fund_id}: {stats['Mean Alpha']:.4f} (±{stats['Std Dev']:.4f})")
    
    print(f"\n📉 BOTTOM PERFORMERS (by mean alpha):")
    bottom_funds = fund_stats.nsmallest(3, 'Mean Alpha')
    for fund_id, stats in bottom_funds.iterrows():
        print(f"  {fund_id}: {stats['Mean Alpha']:.4f} (±{stats['Std Dev']:.4f})")
    
    return df, fund_stats

def create_charts(df, output_dir: str = "predictions/charts/"):
    """Create visualization charts"""
    
    Path(output_dir).mkdir(exist_ok=True)
    plt.style.use('default')
    
    # 1. Distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['ensemble_prediction'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Alpha Predictions')
    plt.xlabel('Ensemble Prediction (Alpha)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/alpha_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot by fund
    plt.figure(figsize=(12, 6))
    df.boxplot(column='ensemble_prediction', by='class_id', ax=plt.gca())
    plt.title('Alpha Predictions by Fund')
    plt.suptitle('')  # Remove default title
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/alpha_by_fund.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Time series if applicable
    if 'month_end' in df.columns and df['month_end'].nunique() > 1:
        plt.figure(figsize=(14, 8))
        for fund_id in df['class_id'].unique():
            fund_data = df[df['class_id'] == fund_id].sort_values('month_end')
            plt.plot(fund_data['month_end'], fund_data['ensemble_prediction'], 
                    marker='o', label=fund_id)
        
        plt.title('Alpha Predictions Over Time')
        plt.xlabel('Date')
        plt.ylabel('Alpha Prediction')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/alpha_timeseries.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📈 Charts saved to: {output_dir}")

if __name__ == "__main__":
    import sys
    
    prediction_file = sys.argv[1] if len(sys.argv) > 1 else "predictions/alpha.parquet"
    
    try:
        df, fund_stats = analyze_predictions(prediction_file)
        
        # Ask if user wants charts
        create_viz = input("\n🎨 Create visualization charts? (y/n): ").lower().startswith('y')
        if create_viz:
            create_charts(df)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nUsage: python compare_predictions.py [prediction_file.parquet]")