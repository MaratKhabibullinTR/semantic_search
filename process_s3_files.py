#!/usr/bin/env python3
"""
Process JSON files from ./data/s3 folder and extract messageid and inline text content.
Creates a combined JSON file with doc_id and content fields.
"""

import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def extract_inline_data_from_xml(xml_content):
    """
    Extract inline text data from XML content.
    Looks for <inlineData contenttype="text/plain"> elements and extracts CDATA content.
    """
    try:
        # Parse the XML
        root = ET.fromstring(xml_content)
        
        # Find inlineData elements with text/plain contenttype
        # Using xpath-like search with namespace handling
        inline_data_elements = []
        
        # Walk through the entire tree to find inlineData elements
        for elem in root.iter():
            if elem.tag.endswith('inlineData') and elem.get('contenttype') == 'text/plain':
                inline_data_elements.append(elem)
        
        # Extract text content from CDATA sections
        text_content = []
        for elem in inline_data_elements:
            if elem.text:
                # Remove CDATA wrapper and clean up the text
                content = elem.text.strip()
                text_content.append(content)
        
        # Join all text content
        return '\n\n'.join(text_content)
        
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return ""
    except Exception as e:
        print(f"Error extracting inline data: {e}")
        return ""


def process_json_file(file_path):
    """
    Process a single JSON file and extract messageid and content.
    Returns a dict with doc_id and content, or None if processing fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract messageid
        messageid = data.get('messageid')
        if not messageid:
            print(f"Warning: No messageid found in {file_path}")
            return None
        
        # Extract XML data
        xml_data = data.get('data')
        if not xml_data:
            print(f"Warning: No data field found in {file_path}")
            return None
        
        # Extract inline text content from XML
        content = extract_inline_data_from_xml(xml_data)
        if not content:
            print(f"Warning: No inline content extracted from {file_path}")
            return None
        
        return {
            "doc_id": messageid,
            "content": content
        }
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def main():
    """
    Main function to process all JSON files and create combined output.
    """
    # Define the data directory
    data_dir = Path("./data/s3")
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # Find all JSON files recursively, excluding those with "index" in the name
    json_files = []
    for json_file in data_dir.rglob("*.json"):
        if "index" not in json_file.name:
            json_files.append(json_file)
    
    print(f"Found {len(json_files)} JSON files to process (excluding index files)")
    
    # Process all files
    combined_data = []
    processed_count = 0
    failed_count = 0
    
    for file_path in json_files:
        print(f"Processing: {file_path}")
        result = process_json_file(file_path)
        
        if result:
            combined_data.append(result)
            processed_count += 1
        else:
            failed_count += 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed to process: {failed_count} files")
    print(f"Total documents in combined file: {len(combined_data)}")
    
    # Save combined data to JSON file
    output_file = "data/combined_s3_data.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nCombined data saved to: {output_file}")
        
        # Show a sample of the first document
        if combined_data:
            print(f"\nSample of first document:")
            print(f"doc_id: {combined_data[0]['doc_id']}")
            print(f"content preview: {combined_data[0]['content'][:200]}...")
            
    except Exception as e:
        print(f"Error saving combined data: {e}")


if __name__ == "__main__":
    main()
