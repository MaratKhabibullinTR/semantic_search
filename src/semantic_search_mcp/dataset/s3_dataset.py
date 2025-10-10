import os
import boto3
import json
import re
from botocore.exceptions import ClientError
from typing import List, Dict, Any, Tuple


BUCKET_NAME = "eap-qa-eu-west-1-source-content"
BASE_PATH = "eap_feed_proxy/"  # The path before feed_id
LOCAL_OUTPUT_DIR = 'data/s3'  # Local directory to save files

FEEDS = ["ucdp"]
TOPICS = ["news_eap", "news_prnus"]
TOP_N = 50  # The number of documents to fetch per topic/feed combination
REGION = "eu-west-1"

# Initialize the S3 client
session = boto3.Session(profile_name='tr-central-preprod')
s3_client = session.client("s3", region_name=REGION)


def is_english_content(data: Dict[str, Any]) -> bool:
    """
    Check if the JSON content is in English.
    
    Args:
        data: The parsed JSON data
        
    Returns:
        bool: True if content is in English, False otherwise
    """
    # Check the language field if it exists
    language = data.get('language', '').lower()
    if language:
        return language.startswith('en')
    
    # If no language field, check headline and other text fields for English patterns
    headline = data.get('headline', '')
    if headline:
        # Simple heuristic: check if headline contains mostly ASCII characters
        # and common English words/patterns
        ascii_ratio = sum(1 for c in headline if ord(c) < 128) / len(headline) if headline else 0
        
        # If less than 80% ASCII characters, likely non-English
        if ascii_ratio < 0.8:
            return False
            
        # Check for common English patterns
        english_patterns = [
            r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b',
            r'\b(is|are|was|were|will|would|could|should)\b',
            r'\b(this|that|these|those)\b'
        ]
        
        english_matches = sum(1 for pattern in english_patterns 
                             if re.search(pattern, headline, re.IGNORECASE))
        
        # If we find at least one English pattern, consider it English
        return english_matches > 0
    
    # Default to True if we can't determine (to avoid false negatives)
    return True


def get_top_n_documents_and_save(
    bucket_name: str, 
    base_path: str, 
    feed_id: str, 
    topic_id: str, 
    top_n: int
) -> int:
    """
    Fetches the top N (most recently modified) JSON documents and saves them to disk.
    
    Returns the count of documents successfully saved.
    """
    
    # 1. Define S3 prefix and local save path
    prefix = f"{base_path.rstrip('/')}/{feed_id}/{topic_id}/"
    local_save_path = os.path.join(LOCAL_OUTPUT_DIR, feed_id, topic_id)
    
    # Create the local directory if it doesn't exist
    os.makedirs(local_save_path, exist_ok=True)
    
    print(f"-> Listing objects for prefix: {prefix}")
    print(f"-> Saving to local path: {local_save_path}")
    
    all_objects = []
    
    # Use Paginator to list all objects under the prefix
    paginator = s3_client.get_paginator('list_objects_v2')
    try:
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].lower().endswith('.json'):
                        all_objects.append(obj)
            
    except ClientError as e:
        print(f"Error listing objects for {prefix}: {e}")
        return 0
    
    if not all_objects:
        print(f"No JSON documents found for {prefix}")
        return 0

    # 2. Sort objects by LastModified date in descending order (newest first)
    sorted_objects = sorted(
        all_objects, 
        key=lambda x: x['LastModified'], 
        reverse=True
    )
    
    # Select the top N keys
    top_n_objects = sorted_objects[:top_n]
    print(f"Found {len(all_objects)} documents. Selecting top {len(top_n_objects)} to download.")

    # 3. Download and save the top N documents
    saved_count = 0
    for obj in top_n_objects:
        s3_key = obj['Key']
        
        # Extract the document's file name from the S3 key
        # e.g., 'some_path/feed_a/topic_1/2025-07-03/uuid.json' -> 'uuid.json'
        # The os.path.basename handles the Unix-style S3 path separators
        file_name = os.path.basename(s3_key) 
        local_file_path = os.path.join(local_save_path, file_name)
        
        try:
            # We can download the file directly to disk without loading into memory first
            s3_client.download_file(
                Bucket=bucket_name, 
                Key=s3_key, 
                Filename=local_file_path
            )
            
            # Optional: Verify it's valid JSON and check if it's English content
            try:
                with open(local_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if the content is in English by examining the language field
                if not is_english_content(data):
                    print(f"   -> SKIPPED: {file_name} contains non-English content")
                    os.remove(local_file_path)  # Remove the non-English file
                    continue
                    
            except UnicodeDecodeError as e:
                print(f"   -> SKIPPED: {file_name} has encoding issues (likely non-English): {e}")
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)  # Remove the problematic file
                continue
            except json.JSONDecodeError as e:
                print(f"   -> WARNING: Downloaded file {file_name} is not valid JSON. Reason: {e}")
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)  # Remove the invalid file
                continue
                
            print(f"   -> SUCCESS: Saved {file_name} to {local_save_path}")
            saved_count += 1
            
        except ClientError as e:
            print(f"   -> ERROR: Failed to download {s3_key}. Reason: {e}")
        except Exception as e:
            print(f"   -> ERROR: Unexpected error processing {file_name}. Reason: {e}")
            if os.path.exists(local_file_path):
                os.remove(local_file_path)  # Remove the problematic file
            
    return saved_count


def download_artifacts_from_s3():
    total_saved_docs = 0
    for feed_id in FEEDS:
        for topic_id in TOPICS:
            print("\n" + "="*50)
            print(f"--- Processing Feed: **{feed_id}**, Topic: **{topic_id}** ---")
            
            count = get_top_n_documents_and_save(
                bucket_name=BUCKET_NAME,
                base_path=BASE_PATH,
                feed_id=feed_id,
                topic_id=topic_id,
                top_n=TOP_N
            )
            total_saved_docs += count
            print(f"Finished. Documents saved for this combo: {count}")

    # --- Final result summary ---
    print("\n" + "="*50)
    print(f"Processing Complete. **{total_saved_docs}** documents were saved in the '{LOCAL_OUTPUT_DIR}' directory.")
    print("="*50)
