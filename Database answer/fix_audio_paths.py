import json
import os
from pathlib import Path

def fix_audio_paths(directory):
    """
    Fix audio paths in all metadata JSON files in the specified directory.
    Changes paths from /content/drive/MyDrive/... to D:\\Database answer\\audio_chunks\\
    """
    # Convert to Path object
    dir_path = Path(directory)
    
    # Counter for tracking changes
    files_processed = 0
    chunks_updated = 0
    
    # Get all JSON files in the directory
    json_files = list(dir_path.glob("*_metadata.json"))
    
    if not json_files:
        print(f"No metadata JSON files found in {directory}")
        return
    
    print(f"Found {len(json_files)} metadata files to process...")
    
    for json_file in json_files:
        try:
            # Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Track if any changes were made
            changed = False
            
            # Update source_audio path if it exists
            if 'source_audio' in data and data['source_audio'].startswith('/content/'):
                data['source_audio'] = data['source_audio'].replace(
                    '/content/drive/MyDrive/working/',
                    'D:\\Database answer\\'
                )
                changed = True
            
            # Update source_transcript path if it exists
            if 'source_transcript' in data and data['source_transcript'].startswith('/content/'):
                data['source_transcript'] = data['source_transcript'].replace(
                    '/content/drive/MyDrive/working/',
                    'D:\\Database answer\\'
                )
                changed = True
            
            # Update audio_path in chunks
            if 'chunks' in data:
                for chunk in data['chunks']:
                    if 'audio_path' in chunk and chunk['audio_path'].startswith('/content/'):
                        # Extract just the filename from the old path
                        old_path = chunk['audio_path']
                        filename = old_path.split('/')[-1]
                        
                        # Create new path
                        chunk['audio_path'] = f"D:\\Database answer\\audio_chunks\\{filename}"
                        changed = True
                        chunks_updated += 1
            
            # Write back if changes were made
            if changed:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                files_processed += 1
                print(f"✓ Updated: {json_file.name}")
        
        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"Files updated: {files_processed}")
    print(f"Chunks updated: {chunks_updated}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Directory containing the metadata files
    metadata_dir = r"D:\Database answer\processed_data_cleaned"
    
    # Verify directory exists
    if not os.path.exists(metadata_dir):
        print(f"Error: Directory not found: {metadata_dir}")
        exit(1)
    
    print(f"Starting audio path fix for: {metadata_dir}")
    print(f"{'='*60}\n")
    
    fix_audio_paths(metadata_dir)
