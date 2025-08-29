#!/usr/bin/env python3
"""
Test the updated OEF/RR extractor with HTML prospectus
"""
from etl.oef_rr_extractor_robust import _extract_rr_fields_from_ixbrl
from pathlib import Path

def test_html_extraction():
    """Test extraction from the sample HTML file"""
    
    # Load the sample HTML document
    sample_file = Path("sample_oef_rr_0000932471.xml")
    if not sample_file.exists():
        print("❌ Sample file not found")
        return
        
    print(f"🔍 Testing extraction from {sample_file}")
    doc_text = sample_file.read_text()
    
    # Extract data
    results = _extract_rr_fields_from_ixbrl(doc_text)
    
    print(f"✅ Extracted {len(results)} records:")
    for i, record in enumerate(results):
        print(f"  {i+1}. {record}")

if __name__ == "__main__":
    test_html_extraction()