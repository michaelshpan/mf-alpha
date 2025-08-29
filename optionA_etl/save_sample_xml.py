#!/usr/bin/env python3
"""
Save a sample OEF/RR XML file for analysis
"""
from etl.oef_rr_extractor_robust import fetch_latest_rr_doc_for_cik_filtered, download_rr_document
import yaml
from pathlib import Path

def save_sample_xml():
    """Download and save a sample OEF/RR XML file"""
    
    # Load pilot config  
    cfg = yaml.safe_load(Path('config/funds_pilot.yaml').read_text())
    regs = cfg.get('registrants', [])
    
    if not regs:
        print("No registrants found in config")
        return
        
    cik = regs[0]['cik']
    print(f"Downloading sample XML for CIK {cik}")
    
    # Try to find a recent OEF/RR filing
    for min_year in [2020, 2015, 2010]:
        hit = fetch_latest_rr_doc_for_cik_filtered(cik, min_year)
        if hit:
            try:
                doc = download_rr_document(cik, hit["accession"], hit["primary_doc"])
                
                # Save to file
                sample_file = Path(f"sample_oef_rr_{cik}.xml")
                with open(sample_file, 'w', encoding='utf-8') as f:
                    f.write(doc)
                
                print(f"✅ Sample XML saved to: {sample_file}")
                print(f"Filing info:")
                print(f"  Accession: {hit['accession']}")
                print(f"  Document: {hit['primary_doc']}")  
                print(f"  Filing date: {hit['filing_date']}")
                print(f"  File size: {len(doc):,} characters")
                
                # Quick preview
                lines = doc.split('\\n')
                print(f"\\n📄 First 20 lines preview:")
                for i, line in enumerate(lines[:20], 1):
                    print(f"{i:2d}: {line[:100]}{'...' if len(line) > 100 else ''}")
                
                return sample_file
                
            except Exception as e:
                print(f"Failed to download from year {min_year}: {e}")
                continue
    
    print("❌ Could not download sample XML file")
    return None

if __name__ == "__main__":
    save_sample_xml()