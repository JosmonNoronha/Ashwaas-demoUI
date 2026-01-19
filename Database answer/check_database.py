#!/usr/bin/env python3
"""
Check Konkani TTS Database Contents
"""

from Konkani_TTS import KonkaniTTS

def check_database():
    """Check what's in the database"""
    print("=" * 60)
    print("Checking Konkani TTS Database")
    print("=" * 60)
    
    tts = KonkaniTTS()
    
    # Get statistics
    stats = tts.get_stats()
    print(f"\n📊 Database Statistics:")
    print(f"   Total Sentences: {stats['total_sentences']}")
    print(f"   Total Words: {stats['total_words']}")
    print(f"   Total Characters: {stats['total_chars']}")
    print(f"   Audio Files: {stats['audio_files']}")
    
    if stats['total_sentences'] == 0:
        print("\n⚠️  Database is empty!")
        print("\n💡 To populate the database, run:")
        print("   python populate_database.py")
        tts.close()
        return
    
    # List first 10 sentences
    print(f"\n📝 First 10 sentences in database:")
    print("-" * 60)
    tts.list_sentences(limit=10)
    
    # Search example
    print("\n🔍 Search Examples:")
    print("\n1. Search for a specific word/phrase:")
    search_term = "नमस्कार"  # You can change this
    results = tts.search(search_term, method='all')
    
    if results['exact']:
        print(f"   ✅ Exact match found for '{search_term}'")
    elif results['similar']:
        print(f"   📊 {len(results['similar'])} similar matches found")
    elif results['contains']:
        print(f"   🔍 {len(results['contains'])} partial matches found")
    else:
        print(f"   ❌ No matches found for '{search_term}'")
    
    tts.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    check_database()
