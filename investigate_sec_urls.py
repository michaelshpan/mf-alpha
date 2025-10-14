#!/usr/bin/env python3
"""
Investigate the correct URLs for SEC N-PORT bulk data downloads
"""

import requests
import re
from bs4 import BeautifulSoup
import json

def investigate_sec_urls():
    """
    Check various URL patterns to find where SEC hosts N-PORT bulk data
    """
    print("="*80)
    print("INVESTIGATING SEC N-PORT BULK DATA URLS")
    print("="*80)
    
    headers = {
        'User-Agent': 'Research User research@university.edu',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
    
    # Test different URL patterns
    test_patterns = [
        # Pattern 1: Original attempted pattern
        {
            'name': 'Original Pattern (form-n-port-data-sets)',
            'base': 'https://www.sec.gov/files/structureddata/data/form-n-port-data-sets',
            'example': 'https://www.sec.gov/files/structureddata/data/form-n-port-data-sets/2023q4_form_nport.zip'
        },
        # Pattern 2: Dera structured data
        {
            'name': 'DERA Structured Data',
            'base': 'https://www.sec.gov/files/dera/data/form-n-port',
            'example': 'https://www.sec.gov/files/dera/data/form-n-port/2023q4_nport.zip'
        },
        # Pattern 3: Edgar bulk data
        {
            'name': 'EDGAR Bulk Data',
            'base': 'https://www.sec.gov/Archives/edgar/quarterly-data',
            'example': 'https://www.sec.gov/Archives/edgar/quarterly-data/2023/QTR4/nport.zip'
        },
        # Pattern 4: Financial statement datasets
        {
            'name': 'Financial Statement Datasets',
            'base': 'https://www.sec.gov/files/dera/data/financial-statement-and-notes-data-sets',
            'example': 'https://www.sec.gov/files/dera/data/financial-statement-and-notes-data-sets/2023q4_notes.zip'
        }
    ]
    
    print("\n1. Testing URL patterns:")
    print("-"*40)
    for pattern in test_patterns:
        print(f"\nTesting: {pattern['name']}")
        print(f"  Example URL: {pattern['example']}")
        
        try:
            response = requests.head(pattern['example'], headers=headers, timeout=5)
            if response.status_code == 200:
                print(f"  ✓ SUCCESS - URL pattern is valid!")
                print(f"  Content-Type: {response.headers.get('Content-Type')}")
                print(f"  Content-Length: {response.headers.get('Content-Length')}")
            else:
                print(f"  ✗ Failed with status: {response.status_code}")
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")
    
    # Check SEC's data page for actual links
    print("\n2. Checking SEC's official data pages:")
    print("-"*40)
    
    data_pages = [
        {
            'name': 'SEC Data Resources',
            'url': 'https://www.sec.gov/data-research/sec-markets-data'
        },
        {
            'name': 'Structured Disclosure',
            'url': 'https://www.sec.gov/structureddata'
        },
        {
            'name': 'DERA Data Library',
            'url': 'https://www.sec.gov/dera/data'
        },
        {
            'name': 'Financial Statement Data Sets',
            'url': 'https://www.sec.gov/dera/data/financial-statement-data-sets'
        }
    ]
    
    for page in data_pages:
        print(f"\nChecking: {page['name']}")
        print(f"  URL: {page['url']}")
        
        try:
            response = requests.get(page['url'], headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for N-PORT related links
                nport_links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    text = link.get_text().strip()
                    
                    if 'nport' in href.lower() or 'nport' in text.lower():
                        nport_links.append({'text': text, 'href': href})
                    elif 'n-port' in href.lower() or 'n-port' in text.lower():
                        nport_links.append({'text': text, 'href': href})
                
                if nport_links:
                    print(f"  ✓ Found {len(nport_links)} N-PORT related links:")
                    for link in nport_links[:5]:  # Show first 5
                        print(f"    - {link['text'][:50]}: {link['href'][:100]}")
                else:
                    print(f"  ✗ No N-PORT links found")
                    
                # Look for quarterly data links
                quarterly_links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if re.search(r'202[0-9]q[1-4]', href.lower()) or 'quarterly' in href.lower():
                        quarterly_links.append(href)
                
                if quarterly_links:
                    print(f"  Found {len(quarterly_links)} quarterly data links")
                    for link in quarterly_links[:3]:
                        print(f"    - {link}")
                        
        except Exception as e:
            print(f"  ✗ Error accessing page: {str(e)[:100]}")
    
    # Check specific financial statement data sets page
    print("\n3. Checking Financial Statement Data Sets (most likely location):")
    print("-"*40)
    
    fs_url = "https://www.sec.gov/dera/data/financial-statement-data-sets"
    try:
        response = requests.get(fs_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all zip file links
            zip_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.zip'):
                    zip_links.append({
                        'text': link.get_text().strip(),
                        'href': href
                    })
            
            print(f"Found {len(zip_links)} ZIP file links")
            
            # Group by pattern
            patterns = {}
            for link in zip_links:
                # Extract pattern from filename
                filename = link['href'].split('/')[-1]
                if '_' in filename:
                    pattern = filename.split('_')[-1].replace('.zip', '')
                    if pattern not in patterns:
                        patterns[pattern] = []
                    patterns[pattern].append(link)
            
            print(f"\nFile patterns found:")
            for pattern, links in patterns.items():
                print(f"  - {pattern}: {len(links)} files")
                if links:
                    print(f"    Example: {links[0]['href']}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    # Check if there's an API endpoint
    print("\n4. Checking for API endpoints:")
    print("-"*40)
    
    api_endpoints = [
        'https://data.sec.gov/api/xbrl/frames/',
        'https://data.sec.gov/submissions/',
        'https://www.sec.gov/Archives/edgar/daily-index/'
    ]
    
    for endpoint in api_endpoints:
        print(f"\nChecking: {endpoint}")
        try:
            response = requests.get(endpoint, headers=headers, timeout=5)
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                print(f"  ✓ Endpoint is accessible")
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")
    
    # Provide recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
Based on the investigation, here are the likely issues and solutions:

1. **URL Pattern Issue**: The SEC likely uses a different URL structure than assumed.
   
2. **Alternative Data Sources**:
   - Use individual N-PORT filings via EDGAR API (already implemented)
   - Use Financial Statement Data Sets (different format)
   - Use commercial data providers
   
3. **Possible Correct Patterns**:
   - Financial Statement datasets: https://www.sec.gov/files/dera/data/financial-statement-data-sets/
   - EDGAR bulk data: https://www.sec.gov/Archives/edgar/
   - Individual filings: https://data.sec.gov/submissions/

4. **Next Steps**:
   - Check SEC's DERA page for actual download links
   - Consider using individual filing approach (current implementation)
   - Contact SEC DERA for bulk data access information
    """)

if __name__ == "__main__":
    investigate_sec_urls()