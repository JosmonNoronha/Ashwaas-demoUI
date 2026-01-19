from flask import Flask, jsonify, request, g, send_file
from flask_cors import CORS
from Konkani_TTS import KonkaniTTS
import sqlite3
import os

app = Flask(__name__)
CORS(app)

# Initialize TTS system (NO DB connection stored here)
tts = KonkaniTTS()

DATABASE_PATH = "konkani_tts.db"


# ---------------- DB CONNECTION (THREAD SAFE) ----------------

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()


# ---------------- API ROUTES ----------------

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    conn = get_db()
    stats = tts.get_stats(conn)
    return jsonify(stats)


@app.route('/api/sentences', methods=['GET'])
def get_sentences():
    """Get all sentences"""
    limit = request.args.get('limit', type=int)
    conn = get_db()
    cursor = conn.cursor()

    query = """
        SELECT id, text, audio_file, word_count
        FROM sentences
        ORDER BY id
    """
    if limit:
        query += " LIMIT ?"
        cursor.execute(query, (limit,))
    else:
        cursor.execute(query)

    results = cursor.fetchall()

    sentences = []
    for row in results:
        sentences.append({
            'id': row['id'],
            'text': row['text'],
            'audio_file': row['audio_file'],
            'word_count': row['word_count']
        })

    return jsonify(sentences)


@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    method = data.get('method', 'all')

    conn = get_db()
    results = tts.search(query, method=method, conn=conn)

    return jsonify({
        'query': results['query'],
        'exact': results['exact'],
        'similar': results['similar'],
        'contains': results['contains']
    })


@app.route('/api/audio/<filename>', methods=['GET'])
def get_audio(filename):
    audio_path = os.path.join(tts.audio_dir, filename)
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/wav')
    return jsonify({'error': 'Audio file not found'}), 404


@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    conn = get_db()
    words = text.strip().split()
    breakdown = []

    for word in words:
        exact_match = tts.find_exact_match(word, conn)

        breakdown.append({
            'word': word,
            'chars': list(word),
            'ascii': [ord(c) for c in word],
            'inDb': bool(exact_match),
            'translation': 'Database entry' if exact_match else 'Unknown',
            'method': 'Direct Playback' if exact_match else 'Concatenation',
            'audio_file': exact_match['audio_file'] if exact_match else None
        })

    return jsonify({'breakdown': breakdown})


@app.route('/api/speak', methods=['POST'])
def speak():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    conn = get_db()

    exact = tts.find_exact_match(text, conn)
    if exact:
        return jsonify({
            'found': True,
            'match_type': 'exact',
            'audio_file': exact['audio_file'],
            'text': exact['text']
        })

    similar = tts.find_similar_sentences(text, threshold=0.7, conn=conn)
    if similar and similar[0]['similarity'] > 0.8:
        return jsonify({
            'found': True,
            'match_type': 'similar',
            'audio_file': similar[0]['audio_file'],
            'text': similar[0]['text'],
            'similarity': similar[0]['similarity']
        })

    return jsonify({
        'found': False,
        'message': 'No suitable match found in database'
    })


# ---------------- SERVER ----------------

if __name__ == '__main__':
    print("🚀 Starting Konkani TTS API Server...")
    app.run(debug=True, port=5000)
