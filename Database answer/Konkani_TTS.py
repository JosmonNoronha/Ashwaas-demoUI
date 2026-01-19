#!/usr/bin/env python3
"""
Konkani Text-to-Speech System
Based on concatenation synthesis with sentence database
"""

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import difflib

try:
    from pydub import AudioSegment
    from pydub.playback import play
    AUDIO_SUPPORT = True
except ImportError:
    AUDIO_SUPPORT = False
    print("⚠️  Install pydub for audio playback: pip install pydub")


class KonkaniTTS:
    def __init__(self, db_path="konkani_tts.db", audio_dir="audio_files"):
        self.db_path = db_path
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(exist_ok=True)
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing sentence mappings"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                audio_file TEXT NOT NULL,
                word_count INTEGER,
                char_count INTEGER,
                added_date TEXT,
                language TEXT DEFAULT 'konkani',
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_text 
            ON sentences(text)
        ''')
        
        self.conn.commit()
        print(f"✅ Database initialized: {self.db_path}")
    
    def add_sentence(self, text, audio_file_path, notes=""):
        """Add a sentence and its audio to the database"""
        if not os.path.exists(audio_file_path):
            print(f"❌ Audio file not found: {audio_file_path}")
            return False
        
        # Copy audio file to audio directory
        audio_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(audio_file_path).name}"
        dest_path = self.audio_dir / audio_filename
        
        try:
            # Copy file
            with open(audio_file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                dst.write(src.read())
            
            # Add to database
            words = text.split()
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO sentences (text, audio_file, word_count, char_count, added_date, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                text.strip(),
                audio_filename,
                len(words),
                len(text.replace(' ', '')),
                datetime.now().isoformat(),
                notes
            ))
            self.conn.commit()
            
            print(f"✅ Added: '{text}' ({len(words)} words)")
            print(f"   Audio: {audio_filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error adding sentence: {e}")
            return False
    
    def find_exact_match(self, text):
        """Find exact sentence match"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, text, audio_file, word_count 
            FROM sentences 
            WHERE text = ?
        ''', (text.strip(),))
        
        result = cursor.fetchone()
        if result:
            return {
                'id': result[0],
                'text': result[1],
                'audio_file': result[2],
                'word_count': result[3],
                'match_type': 'exact'
            }
        return None
    
    def find_similar_sentences(self, text, threshold=0.6):
        """Find similar sentences using fuzzy matching"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, text, audio_file, word_count FROM sentences')
        all_sentences = cursor.fetchall()
        
        matches = []
        query_lower = text.lower().strip()
        
        for row in all_sentences:
            db_text = row[1].lower()
            
            # Calculate similarity ratio
            ratio = difflib.SequenceMatcher(None, query_lower, db_text).ratio()
            
            if ratio >= threshold:
                matches.append({
                    'id': row[0],
                    'text': row[1],
                    'audio_file': row[2],
                    'word_count': row[3],
                    'similarity': ratio,
                    'match_type': 'similar'
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches
    
    def find_partial_matches(self, text):
        """Find sentences containing the query text"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, text, audio_file, word_count 
            FROM sentences 
            WHERE text LIKE ?
        ''', (f'%{text.strip()}%',))
        
        results = cursor.fetchall()
        matches = []
        
        for row in results:
            matches.append({
                'id': row[0],
                'text': row[1],
                'audio_file': row[2],
                'word_count': row[3],
                'match_type': 'contains'
            })
        
        return matches
    
    def search(self, text, method='all'):
        """
        Search for matching sentences
        Methods: 'exact', 'similar', 'contains', 'all'
        """
        results = {
            'query': text,
            'exact': None,
            'similar': [],
            'contains': []
        }
        
        if method in ['exact', 'all']:
            results['exact'] = self.find_exact_match(text)
        
        if method in ['similar', 'all']:
            results['similar'] = self.find_similar_sentences(text)
        
        if method in ['contains', 'all']:
            results['contains'] = self.find_partial_matches(text)
        
        return results
    
    def play_sentence(self, sentence_id=None, audio_file=None):
        """Play audio for a sentence"""
        if not AUDIO_SUPPORT:
            print("❌ Audio playback not available. Install pydub: pip install pydub")
            return False
        
        if audio_file is None and sentence_id is not None:
            cursor = self.conn.cursor()
            cursor.execute('SELECT audio_file FROM sentences WHERE id = ?', (sentence_id,))
            result = cursor.fetchone()
            if result:
                audio_file = result[0]
            else:
                print(f"❌ Sentence ID {sentence_id} not found")
                return False
        
        audio_path = self.audio_dir / audio_file
        
        if not audio_path.exists():
            print(f"❌ Audio file not found: {audio_path}")
            return False
        
        try:
            print(f"🔊 Playing: {audio_file}")
            audio = AudioSegment.from_file(audio_path)
            play(audio)
            return True
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
            return False
    
    def speak(self, text):
        """Main TTS function - find and play matching sentence"""
        print(f"\n🔍 Searching for: '{text}'")
        
        # Try exact match first
        exact = self.find_exact_match(text)
        if exact:
            print(f"✅ Exact match found!")
            print(f"   Text: {exact['text']}")
            print(f"   Words: {exact['word_count']}")
            self.play_sentence(audio_file=exact['audio_file'])
            return True
        
        # Try similar matches
        similar = self.find_similar_sentences(text, threshold=0.7)
        if similar:
            print(f"📊 Found {len(similar)} similar sentence(s):")
            for i, match in enumerate(similar[:3], 1):
                print(f"\n   {i}. {match['text']}")
                print(f"      Similarity: {match['similarity']*100:.1f}%")
            
            if similar[0]['similarity'] > 0.9:
                print(f"\n🔊 Playing closest match...")
                self.play_sentence(audio_file=similar[0]['audio_file'])
                return True
            else:
                print(f"\n⚠️  No close matches. Use concatenation for: {text}")
                return False
        
        print(f"❌ No matches found. Use concatenation synthesis.")
        return False
    
    def list_sentences(self, limit=None):
        """List all sentences in database"""
        cursor = self.conn.cursor()
        query = 'SELECT id, text, audio_file, word_count, added_date FROM sentences ORDER BY added_date DESC'
        
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        print(f"\n📚 Database contains {len(results)} sentence(s):\n")
        for row in results:
            print(f"ID: {row[0]}")
            print(f"Text: {row[1]}")
            print(f"Words: {row[3]} | Audio: {row[2]}")
            print(f"Added: {row[4][:10]}")
            print("-" * 50)
        
        return results
    
    def get_stats(self):
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*), SUM(word_count), SUM(char_count) FROM sentences')
        stats = cursor.fetchone()
        
        return {
            'total_sentences': stats[0] or 0,
            'total_words': stats[1] or 0,
            'total_chars': stats[2] or 0,
            'audio_files': len(list(self.audio_dir.glob('*')))
        }
    
    def export_database(self, output_file="konkani_tts_export.json"):
        """Export database to JSON"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM sentences')
        
        columns = [desc[0] for desc in cursor.description]
        data = []
        
        for row in cursor.fetchall():
            data.append(dict(zip(columns, row)))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Exported {len(data)} sentences to {output_file}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Example usage"""
    print("=" * 60)
    print("कोंकणी Text-to-Speech System")
    print("Konkani TTS with Sentence Database")
    print("=" * 60)
    
    tts = KonkaniTTS()
    
    # Load data from processed_data_cleaned and audio_chunks
    print("\n--- Loading Data from processed_data_cleaned ---")
    
    metadata_dir = Path("D:/Database answer/processed_data_cleaned")
    sample_data = []
    
    # Read all metadata JSON files
    if metadata_dir.exists():
        json_files = list(metadata_dir.glob("*_metadata.json"))
        print(f"Found {len(json_files)} metadata files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract chunks
                if 'chunks' in data:
                    for chunk in data['chunks']:
                        text = chunk.get('text', '')
                        audio_path = chunk.get('audio_path', '')
                        chunk_index = chunk.get('chunk_index', 0)
                        
                        # Use the audio path from metadata (should point to audio_chunks)
                        if text and audio_path and os.path.exists(audio_path):
                            notes = f"Chunk {chunk_index} from {json_file.stem}"
                            sample_data.append((text, audio_path, notes))
            
            except Exception as e:
                print(f"Error reading {json_file.name}: {e}")
        
        print(f"Loaded {len(sample_data)} text-audio pairs")
    else:
        print(f"⚠️  Metadata directory not found: {metadata_dir}")
        sample_data = []
    
    print("\n💡 To add sentences to database:")
    print("   for text, audio, notes in sample_data:")
    print("       tts.add_sentence(text, audio, notes)")
    
    # Show stats
    stats = tts.get_stats()
    print(f"\n📊 Database Stats:")
    print(f"   Total Sentences: {stats['total_sentences']}")
    print(f"   Total Words: {stats['total_words']}")
    print(f"   Total Characters: {stats['total_chars']}")
    print(f"   Audio Files: {stats['audio_files']}")
    
    # Example: Search and speak
    print("\n--- Example Usage ---")
    print("\n1. Exact match:")
    print("   tts.speak('नमस्कार')")
    
    print("\n2. Search for similar:")
    print("   results = tts.search('नमस्कार', method='all')")
    
    print("\n3. List all sentences:")
    print("   tts.list_sentences(limit=10)")
    
    print("\n4. Export database:")
    print("   tts.export_database('my_konkani_db.json')")
    
    tts.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()