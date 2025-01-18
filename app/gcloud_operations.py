
from google.cloud import storage
import os
import glob
from google.cloud import storage

from pathlib import Path



def download_file(bucket_name, source_file_name,  destination_file_name):
    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    
    blob = bucket.blob(source_file_name)
    # Download the file to a destination
    blob.download_to_filename(destination_file_name)