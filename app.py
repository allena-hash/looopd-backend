from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import similarity

app = Flask(__name__)
CORS(app)

# ── Supabase config ──────────────────────────────────────────
SB_URL = os.environ.get("SUPABASE_URL", "https://icfgkrinhmiebmwwqarj.supabase.co")
SB_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImljZmdrcmluaG1pZWJtd3dxYXJqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzUxMDM4NDUsImV4cCI6MjA5MDY3OTg0NX0.v3rOuoZBFqCb6-gYN-MI0uJGBJr4jws6BQk431Ic-XQ")

SB_HEADERS = {
    "apikey": SB_KEY,
    "Authorization": f"Bearer {SB_KEY}",
    "Content-Type": "application/json"
}

def sb_get(path):
    r = requests.get(f"{SB_URL}/rest/v1/{path}", headers=SB_HEADERS)
    r.raise_for_status()
    return r.json()


# ── Health ───────────────────────────────────────────────────
@app.route("/health")
def health():
    return {"status": "ok"}


# ── Emotion detection ────────────────────────────────────────
@app.route("/detect-emotion", methods=["POST"])
def detect_emotion():
    """
    Takes a transcript and returns emotion + detail.
    POST { "text": "I feel so lonely today..." }
    Returns { "emotion": "lonely", "detail": "quietly-lonely", "scores": {...} }
    """
    body = request.get_json()
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"emotion": "neutral", "detail": "quietly-present", "scores": {}})

    emotion, detail, scores = similarity.detect_emotion(text)
    return jsonify({"emotion": emotion, "detail": detail, "scores": scores})


# ── Similarity ───────────────────────────────────────────────
@app.route("/similar", methods=["POST"])
def find_similar():
    """
    Takes a transcript + question_id + exclude_id, returns top N similar responses.
    POST {
        "text": "I feel really sad...",
        "question_id": "uuid",
        "exclude_id": "uuid",   # the response just recorded (exclude it)
        "n": 2                  # how many similar to return
    }
    Returns { "similar": [ {response row}, ... ], "scores": [0.87, 0.72] }
    """
    body = request.get_json()
    text = body.get("text", "").strip()
    question_id = body.get("question_id")
    exclude_id = body.get("exclude_id")
    n = int(body.get("n", 2))

    if not text or not question_id:
        return jsonify({"error": "text and question_id required"}), 400

    # Fetch all responses for this question from Supabase
    path = f"responses?question_id=eq.{question_id}&select=*"
    if exclude_id:
        path += f"&id=neq.{exclude_id}"

    try:
        responses = sb_get(path)
    except Exception as e:
        return jsonify({"error": f"Supabase error: {str(e)}"}), 500

    if not responses:
        return jsonify({"similar": [], "scores": []})

    # Get transcripts for similarity calculation
    candidates_text = [r.get("transcript", "") for r in responses]

    # Find top N most similar using sentence embeddings
    top_indices, scores = similarity.top_n_similar(text, candidates_text, n=n)

    similar = [responses[i] for i in top_indices]
    return jsonify({
        "similar": similar,
        "scores": [round(float(scores[i]), 3) for i in range(len(top_indices))]
    })


# ── Emotion + Similarity combined ────────────────────────────
@app.route("/process", methods=["POST"])
def process():
    """
    One-shot endpoint: detect emotion AND find similar responses.
    POST {
        "text": "transcript here",
        "question_id": "uuid",
        "exclude_id": "uuid"
    }
    Returns {
        "emotion": "sad",
        "detail": "deeply-sad",
        "similar": [...],
        "scores": [...]
    }
    """
    body = request.get_json()
    text = body.get("text", "").strip()
    question_id = body.get("question_id")
    exclude_id = body.get("exclude_id")

    if not text or not question_id:
        return jsonify({"error": "text and question_id required"}), 400

    # Detect emotion
    emotion, detail, _ = similarity.detect_emotion(text)

    # Fetch responses
    path = f"responses?question_id=eq.{question_id}&select=*"
    if exclude_id:
        path += f"&id=neq.{exclude_id}"

    try:
        responses = sb_get(path)
    except Exception as e:
        return jsonify({"emotion": emotion, "detail": detail, "similar": [], "scores": [], "error": str(e)})

    similar = []
    scores = []

    if responses:
        candidates_text = [r.get("transcript", "") for r in responses]
        top_indices, top_scores = similarity.top_n_similar(text, candidates_text, n=2)
        similar = [responses[i] for i in top_indices]
        scores = [round(float(top_scores[i]), 3) for i in range(len(top_indices))]

    return jsonify({
        "emotion": emotion,
        "detail": detail,
        "similar": similar,
        "scores": scores
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
