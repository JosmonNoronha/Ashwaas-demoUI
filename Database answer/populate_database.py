#!/usr/bin/env python3
"""
Populate Konkani TTS Database with data from processed_data_cleaned
"""

from Konkani_TTS import KonkaniTTS
from pathlib import Path
import json
import os

def populate_database():
    """Load all sentences from processed_data_cleaned into database"""
    print("=" * 60)
    print("Populating Konkani TTS Database")
    print("=" * 60)
    
    tts = KonkaniTTS()
    
    # Load data from processed_data_cleaned
    metadata_dir = Path("D:/Database answer/processed_data_cleaned")
    
    if not metadata_dir.exists():
        print(f"❌ Metadata directory not found: {metadata_dir}")
        return
    
    json_files = list(metadata_dir.glob("*_metadata.json"))
    print(f"\nFound {len(json_files)} metadata files")
    
    total_added = 0
    total_skipped = 0
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract chunks
            if 'chunks' in data:
                for chunk in data['chunks']:
                    text = chunk.get('text', '')
                    audio_path = chunk.get('audio_path', '')
                    chunk_index = chunk.get('chunk_index', 0)
                    
                    if text and audio_path:
                        # Check if audio file exists
                        if os.path.exists(audio_path):
                            notes = f"Chunk {chunk_index} from {json_file.stem}"
                            
                            # Add to database
                            if tts.add_sentence(text, audio_path, notes):
                                total_added += 1
                            else:
                                total_skipped += 1
                        else:
                            print(f"   ⚠️  Audio not found: {audio_path}")
                            total_skipped += 1
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Successfully added: {total_added} sentences")
    print(f"⚠️  Skipped: {total_skipped} sentences")
    print("=" * 60)
    
    # Show final stats
    stats = tts.get_stats()
    print(f"\n📊 Final Database Stats:")
    print(f"   Total Sentences: {stats['total_sentences']}")
    print(f"   Total Words: {stats['total_words']}")
    print(f"   Total Characters: {stats['total_chars']}")
    print(f"   Audio Files: {stats['audio_files']}")
    
    tts.close()
    print("\n✅ Database populated!")

if __name__ == "__main__":
    populate_database()
