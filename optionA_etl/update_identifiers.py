#!/usr/bin/env python3
"""
Standalone script to update the identifier mapping cache.

Usage:
    python update_identifiers.py [--cache-dir ./cache] [--force]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from etl.identifier_mapper import IdentifierMapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Update mutual fund identifier mapping cache"
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./cache',
        help='Directory for cache storage (default: ./cache)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force update even if cache is valid'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show cache statistics only'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize mapper
        mapper = IdentifierMapper(cache_dir=args.cache_dir)
        
        if args.stats:
            # Show statistics
            stats = mapper.get_stats()
            print("\n=== Identifier Cache Statistics ===")
            for key, value in stats.items():
                print(f"{key:20}: {value}")
        else:
            # Update cache
            log.info("Starting identifier cache update")
            mapper.update_cache(force=args.force)
            
            # Show updated stats
            stats = mapper.get_stats()
            print("\n=== Cache Updated Successfully ===")
            print(f"Records: {stats['total_records']:,}")
            print(f"CIKs: {stats['unique_ciks']:,}")
            print(f"Series: {stats['unique_series']:,}")
            print(f"Classes: {stats['unique_classes']:,}")
            print(f"Tickers: {stats['unique_tickers']:,}")
            
    except Exception as e:
        log.error(f"Failed to update identifier cache: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()