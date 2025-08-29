#!/usr/bin/env python3
"""
Convert parquet files to other formats for easier viewing
"""
import pandas as pd
import sys
from pathlib import Path

def convert_parquet(input_file: str, output_format: str = "csv"):
    """Convert parquet to other formats"""
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    print(f"🔄 Converting {input_file} to {output_format.upper()}...")
    
    # Load data
    df = pd.read_parquet(input_file)
    print(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Generate output filename
    output_file = input_path.with_suffix(f".{output_format}")
    
    try:
        if output_format.lower() == "csv":
            df.to_csv(output_file, index=False)
            
        elif output_format.lower() == "excel" or output_format.lower() == "xlsx":
            output_file = input_path.with_suffix(".xlsx")
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
                
                # Add summary sheet if predictions
                if 'ensemble_prediction' in df.columns and 'class_id' in df.columns:
                    summary = df.groupby('class_id')['ensemble_prediction'].agg([
                        'count', 'mean', 'std', 'min', 'max'
                    ]).round(4)
                    summary.to_excel(writer, sheet_name='Summary')
                    
        elif output_format.lower() == "json":
            df.to_json(output_file, orient='records', date_format='iso', indent=2)
            
        elif output_format.lower() == "html":
            html_content = f"""
            <html>
            <head><title>Data from {input_file}</title></head>
            <body>
            <h1>Data from {input_file}</h1>
            <p>Rows: {len(df):,} | Columns: {len(df.columns)}</p>
            {df.to_html(max_rows=1000, table_id='data-table')}
            </body>
            </html>
            """
            with open(output_file.with_suffix('.html'), 'w') as f:
                f.write(html_content)
            output_file = output_file.with_suffix('.html')
            
        else:
            print(f"❌ Unsupported format: {output_format}")
            return
            
        print(f"✅ Converted to: {output_file}")
        print(f"📁 File size: {output_file.stat().st_size / 1024:.1f} KB")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")

def batch_convert(directory: str = ".", formats: list = ["csv", "excel"]):
    """Convert all parquet files in directory"""
    
    parquet_files = list(Path(directory).glob("*.parquet"))
    
    if not parquet_files:
        print(f"❌ No parquet files found in: {directory}")
        return
    
    print(f"🔍 Found {len(parquet_files)} parquet files")
    
    for pq_file in parquet_files:
        print(f"\n📄 Processing: {pq_file.name}")
        for fmt in formats:
            convert_parquet(str(pq_file), fmt)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Convert parquet files to other formats for easier viewing")
        print("\nUsage:")
        print("  python convert_parquet.py <file.parquet> [format]")
        print("  python convert_parquet.py --batch [directory] [formats...]")
        print("\nFormats: csv, excel, json, html")
        print("\nExamples:")
        print("  python convert_parquet.py predictions/alpha.parquet csv")
        print("  python convert_parquet.py predictions/alpha.parquet excel")
        print("  python convert_parquet.py --batch predictions/ csv excel")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        directory = sys.argv[2] if len(sys.argv) > 2 else "."
        formats = sys.argv[3:] if len(sys.argv) > 3 else ["csv", "excel"]
        batch_convert(directory, formats)
    else:
        input_file = sys.argv[1]
        output_format = sys.argv[2] if len(sys.argv) > 2 else "csv"
        convert_parquet(input_file, output_format)