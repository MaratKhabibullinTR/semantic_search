#!/usr/bin/env python3
"""
Script to generate qrels.jsonl from combined_s3_data.json for semantic search validation.
Creates relevant search queries for each document based on content analysis.
"""

import json
import re
from typing import List, Dict, Set


def extract_key_phrases(content: str) -> List[str]:
    """Extract key phrases and topics from document content."""
    # Remove URLs and formatting
    clean_content = re.sub(r'https?://\S+', '', content)
    clean_content = re.sub(r'[^\w\s]', ' ', clean_content)
    
    # Common business/news keywords to look for
    business_keywords = [
        'revenue', 'growth', 'profit', 'earnings', 'investment', 'market', 'sales',
        'acquisition', 'merger', 'IPO', 'stock', 'share', 'dividend', 'finance',
        'technology', 'innovation', 'product', 'service', 'customer', 'client',
        'industry', 'sector', 'competition', 'partnership', 'agreement', 'deal'
    ]
    
    # Extract company names (words in all caps or starting with capital letters)
    company_pattern = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*(?:\s+(?:Inc|Corp|Ltd|LLC|Co)\b)?'
    companies = re.findall(company_pattern, clean_content)
    companies = [c.strip() for c in companies if len(c) > 2 and not c.lower() in ['the', 'and', 'for', 'with']]
    
    # Extract locations (cities, countries)
    location_pattern = r'\b[A-Z][a-zA-Z]+(?:,\s*[A-Z][a-zA-Z]+)*\b'
    locations = re.findall(location_pattern, clean_content)
    
    # Find business keywords in content
    found_keywords = []
    content_lower = clean_content.lower()
    for keyword in business_keywords:
        if keyword in content_lower:
            found_keywords.append(keyword)
    
    # Extract numbers with context (revenue figures, percentages, etc.)
    number_contexts = re.findall(r'(\w+\s*)?[\$]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|percent|%))?', content)
    
    return {
        'companies': companies[:3],  # Top 3 companies
        'locations': locations[:2],   # Top 2 locations
        'keywords': found_keywords[:5],  # Top 5 keywords
        'has_financial_data': ('$' in content or '%' in content or 'million' in content or 'billion' in content)
    }


def generate_queries_for_document(doc_id: str, content: str) -> List[str]:
    """Generate relevant search queries for a document."""
    key_info = extract_key_phrases(content)
    queries = []
    
    # Company-based queries
    for company in key_info['companies'][:2]:  # Top 2 companies
        if len(company.split()) <= 3:  # Avoid very long company names
            queries.append(f"{company} news")
            queries.append(f"{company} announcement")
    
    # Location-based queries
    for location in key_info['locations'][:1]:  # Top location
        if len(location.split()) <= 2:
            queries.append(f"{location} business news")
    
    # Keyword-based queries
    for keyword in key_info['keywords'][:3]:  # Top 3 keywords
        queries.append(keyword)
        if key_info['has_financial_data']:
            queries.append(f"{keyword} financial results")
    
    # Generic queries based on content type
    if 'battery' in content.lower() or 'energy' in content.lower():
        queries.append("energy storage technology")
        queries.append("battery industry news")
    
    if 'restaurant' in content.lower() or 'food' in content.lower():
        queries.append("restaurant industry")
        queries.append("food service news")
    
    if 'AI' in content or 'artificial intelligence' in content.lower():
        queries.append("AI technology")
        queries.append("artificial intelligence news")
    
    if 'meeting' in content.lower() and 'shareholder' in content.lower():
        queries.append("shareholder meeting results")
        queries.append("corporate governance")
    
    # Remove duplicates and empty queries
    queries = list(set([q.strip() for q in queries if q.strip() and len(q.strip()) > 2]))
    
    return queries[:5]  # Limit to top 5 queries per document


def main():
    """Main function to process documents and generate qrels."""
    
    # Load the combined document data
    with open('combined_s3_data.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"Processing {len(documents)} documents...")
    
    # Generate queries for each document
    all_queries = {}  # query -> set of relevant doc_ids
    
    for doc in documents:
        doc_id = doc['doc_id']
        content = doc['content']
        
        # Generate queries for this document
        queries = generate_queries_for_document(doc_id, content)
        
        print(f"Generated {len(queries)} queries for doc {doc_id[:20]}...")
        
        # Add to global query mapping
        for query in queries:
            if query not in all_queries:
                all_queries[query] = set()
            all_queries[query].add(doc_id)
    
    # Convert to qrels format and write to file
    qrels_data = []
    for query, doc_ids in all_queries.items():
        if len(doc_ids) > 0:  # Only include queries with relevant documents
            qrels_entry = {
                "query": query,
                "relevant_doc_ids": list(doc_ids)
            }
            qrels_data.append(qrels_entry)
    
    # Write to qrels.jsonl file
    with open('qrels.jsonl', 'w', encoding='utf-8') as f:
        for entry in qrels_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\nGenerated {len(qrels_data)} unique queries")
    print(f"Updated qrels.jsonl with validation dataset")
    
    # Print some sample queries
    print("\nSample queries generated:")
    for i, entry in enumerate(qrels_data[:10]):
        print(f"{i+1}. '{entry['query']}' -> {len(entry['relevant_doc_ids'])} relevant docs")


if __name__ == "__main__":
    main()
