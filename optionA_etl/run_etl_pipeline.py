#!/usr/bin/env python3
"""
Main script to run the redesigned mutual fund ETL pipeline.

Usage:
    python run_etl_pipeline.py --start 2020-01-01 --end 2023-12-31 [options]
    
Options:
    --start DATE          Start date (YYYY-MM-DD) for requested data
    --end DATE           End date (YYYY-MM-DD) for requested data  
    --config FILE        Path to fund configuration (default: config/funds_pilot.yaml)
    --auto-dedupe        Automatically deduplicate with default strategy
    --dedupe-strategy    Strategy for auto-deduplication (latest|earliest|average|sec_priority)
    --cache-dir DIR      Cache directory (default: ./cache)
    --output-dir DIR     Output directory (default: ./outputs)
    --update-identifiers Update identifier cache before running
    --verbose           Enable verbose logging
    
Examples:
    # Run with manual duplicate resolution
    python run_etl_pipeline.py --start 2020-01-01 --end 2023-12-31
    
    # Run with automatic deduplication using 'latest' strategy
    python run_etl_pipeline.py --start 2020-01-01 --end 2023-12-31 --auto-dedupe --dedupe-strategy latest
    
    # Update identifier cache first, then run
    python run_etl_pipeline.py --start 2020-01-01 --end 2023-12-31 --update-identifiers
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables if not already set
if 'SEC_USER_AGENT' not in os.environ:
    os.environ['SEC_USER_AGENT'] = 'Research Contact research@example.com'

from etl.etl_orchestrator import ETLOrchestrator
from etl.identifier_mapper import IdentifierMapper

# Configure logging
def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"etl_pipeline_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    
    # Reduce verbosity of some libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return log_file


def validate_date(date_string: str) -> str:
    """Validate date format."""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return date_string
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_string}. Use YYYY-MM-DD")


def main():
    """Main entry point for ETL pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the mutual fund ETL pipeline with 36-month lookback for factor regressions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Examples:')[1] if 'Examples:' in __doc__ else ""
    )
    
    # Required arguments
    parser.add_argument(
        '--start',
        type=validate_date,
        required=True,
        help='Start date for requested data (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=validate_date,
        required=True,
        help='End date for requested data (YYYY-MM-DD)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config',
        type=str,
        default='config/funds_pilot.yaml',
        help='Path to fund configuration file'
    )
    parser.add_argument(
        '--auto-dedupe',
        action='store_true',
        help='Automatically deduplicate data (skip manual consultation)'
    )
    parser.add_argument(
        '--dedupe-strategy',
        type=str,
        choices=['latest', 'earliest', 'average', 'sec_priority'],
        default='latest',
        help='Strategy for automatic deduplication'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./cache',
        help='Directory for caching data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Directory for output files'
    )
    parser.add_argument(
        '--update-identifiers',
        action='store_true',
        help='Update identifier cache before running'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.verbose)
    log = logging.getLogger(__name__)
    
    log.info("="*70)
    log.info("MUTUAL FUND ETL PIPELINE - REDESIGNED VERSION")
    log.info("="*70)
    log.info(f"Start Date: {args.start} (+ 36 months lookback)")
    log.info(f"End Date: {args.end}")
    log.info(f"Configuration: {args.config}")
    log.info(f"Log File: {log_file}")
    log.info("="*70)
    
    try:
        # Update identifier cache if requested
        if args.update_identifiers:
            log.info("\n=== Updating Identifier Cache ===")
            mapper = IdentifierMapper(cache_dir=args.cache_dir)
            mapper.update_cache(force=True)
            stats = mapper.get_stats()
            log.info(f"Cache updated: {stats['unique_classes']} classes, "
                    f"{stats['unique_tickers']} tickers")
        
        # Initialize orchestrator
        log.info("\n=== Initializing ETL Orchestrator ===")
        orchestrator = ETLOrchestrator(
            config_path=args.config,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir
        )
        
        # Run ETL pipeline
        log.info("\n=== Starting ETL Pipeline ===")
        log.info(f"Auto-deduplication: {'Yes' if args.auto_dedupe else 'No'}")
        if args.auto_dedupe:
            log.info(f"Deduplication strategy: {args.dedupe_strategy}")
        
        # Execute pipeline
        result_df = orchestrator.run_etl(
            start_date=args.start,
            end_date=args.end,
            auto_dedupe=args.auto_dedupe,
            dedupe_strategy=args.dedupe_strategy
        )
        
        # Handle manual duplicate resolution if needed
        if result_df is None and not args.auto_dedupe:
            print("\n" + "="*70)
            print("MANUAL DUPLICATE RESOLUTION REQUIRED")
            print("="*70)
            print("\nDuplicates were detected. You have three options:")
            print("\n1. Re-run with automatic deduplication:")
            print(f"   python run_etl_pipeline.py --start {args.start} --end {args.end} "
                  f"--auto-dedupe --dedupe-strategy latest")
            print("\n2. Use interactive resolution (in Python):")
            print("   >>> from etl.etl_orchestrator import ETLOrchestrator")
            print(f"   >>> orch = ETLOrchestrator('{args.config}')")
            print(f"   >>> orch.run_etl('{args.start}', '{args.end}')")
            print("   >>> # Review duplicate report")
            print("   >>> orch.resolve_duplicates({'returns': 'latest', 'flows': 'sec_priority'})")
            print("\n3. Edit this script to specify custom resolution per data source")
            print("\n" + "="*70)
            
            # Save intermediate state
            state_file = Path(args.output_dir) / f"etl_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            import pickle
            with open(state_file, 'wb') as f:
                pickle.dump({
                    'orchestrator': orchestrator,
                    'args': args
                }, f)
            print(f"\nIntermediate state saved to: {state_file}")
            print("You can load and continue processing later")
            
            sys.exit(1)
        
        if result_df is not None:
            log.info("\n" + "="*70)
            log.info("ETL PIPELINE COMPLETED SUCCESSFULLY")
            log.info("="*70)
            log.info(f"Output records: {len(result_df)}")
            log.info(f"Unique funds: {result_df['class_id'].nunique()}")
            log.info(f"Output directory: {args.output_dir}")
            
            # Show output files
            output_files = list(Path(args.output_dir).glob("*.parquet"))
            if output_files:
                latest_file = max(output_files, key=lambda f: f.stat().st_mtime)
                log.info(f"Latest output file: {latest_file}")
        
    except KeyboardInterrupt:
        log.warning("\nPipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


def interactive_duplicate_resolution():
    """
    Helper function for interactive duplicate resolution.
    Can be called from Python REPL after initial run.
    """
    import pickle
    
    print("Loading saved ETL state...")
    
    # Find most recent state file
    state_files = list(Path("./outputs").glob("etl_state_*.pkl"))
    if not state_files:
        print("No saved state found. Run the pipeline first.")
        return
    
    latest_state = max(state_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading from: {latest_state}")
    
    with open(latest_state, 'rb') as f:
        state = pickle.load(f)
    
    orchestrator = state['orchestrator']
    args = state['args']
    
    print("\nDuplicate report:")
    print(orchestrator.raw_data.get('_duplicate_report', 'No duplicates found'))
    
    print("\nEnter resolution strategies (or 'quit' to exit):")
    resolutions = {}
    
    for source in ['returns', 'flows', 'tna', 'expense_turnover', 'manager_tenure']:
        if source in orchestrator.raw_data and not orchestrator.raw_data[source].empty:
            strategy = input(f"{source} [latest/earliest/average/sec_priority]: ").strip()
            if strategy == 'quit':
                return
            if strategy in ['latest', 'earliest', 'average', 'sec_priority']:
                resolutions[source] = strategy
    
    print(f"\nApplying resolutions: {resolutions}")
    result = orchestrator.resolve_duplicates(resolutions)
    
    if result is not None:
        print("\nETL completed successfully!")
        print(f"Output: {len(result)} records")
    
    return result


if __name__ == "__main__":
    main()