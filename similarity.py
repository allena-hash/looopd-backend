from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

_model = None


print("Loading sentence transformer model...")
_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully")

def _get_model():
    return _model

def top_n_similar(query: str, candidates: list[str], n: int = 2) -> tuple[list[int], list[float]]:
    """
    Returns indices and scores of the top N most similar candidates to the query.
    Filters out empty candidates.
    """
    if not candidates:
        return [], []

    # Filter empty strings
    valid = [(i, c) for i, c in enumerate(candidates) if c and c.strip()]
    if not valid:
        return [], []

    indices, texts = zip(*valid)
    model = _get_model()
    embeddings = model.encode([query] + list(texts))
    query_vec = embeddings[0:1]
    candidate_vecs = embeddings[1:]
    scores = cosine_similarity(query_vec, candidate_vecs)[0]

    # Get top N
    n = min(n, len(scores))
    top_n = np.argsort(scores)[::-1][:n]
    top_indices = [indices[i] for i in top_n]
    top_scores = [float(scores[i]) for i in top_n]

    return top_indices, top_scores


def most_similar(query: str, candidates: list[str]) -> tuple[int, float]:
    """
    Original function — returns single best match index and score.
    """
    indices, scores = top_n_similar(query, candidates, n=1)
    if not indices:
        return 0, 0.0
    return indices[0], scores[0]


# ── Emotion detection using sentence embeddings ──────────────────────
# We embed the input text and compare it to prototype sentences
# for each emotion. The closest emotion prototype wins.
# This is far more accurate than keyword matching.

EMOTION_PROTOTYPES = {
    "happy": [
        "I feel so happy and joyful today",
        "Everything is going great, I am having such a good time",
        "I am really enjoying this, it makes me smile",
        "I feel grateful and blessed, life is good",
        "I am so excited and thrilled about this",
    ],
    "sad": [
        "I feel so sad and down today",
        "I have been crying and I feel really low",
        "Everything feels hopeless and I am hurting inside",
        "I feel heartbroken and devastated",
        "Nothing is going right and I feel terrible",
    ],
    "lonely": [
        "I feel so alone and nobody understands me",
        "I have no one to talk to and I feel invisible",
        "I feel completely isolated and disconnected from everyone",
        "Nobody cares about me and I feel forgotten",
        "I feel like a stranger even around people I know",
    ],
    "anxious": [
        "I feel so anxious and worried about everything",
        "My mind is racing and I cannot stop overthinking",
        "I feel scared and nervous, something bad might happen",
        "I am panicking and feel so stressed out",
        "I cannot sleep because I keep worrying about things",
    ],
    "overwhelmed": [
        "I feel completely overwhelmed, there is too much going on",
        "I cannot cope with everything at once, I am drowning",
        "I am exhausted and burned out from all the pressure",
        "Everything is piling up and I am at my breaking point",
        "I have so much to do and I do not know where to start",
    ],
    "angry": [
        "I am so angry and furious about this",
        "I feel betrayed and outraged, this is so unfair",
        "I hate this situation and I am really mad",
        "I am fed up and cannot take this anymore",
        "I feel so much rage and resentment",
    ],
    "frustrated": [
        "I feel so frustrated, nothing is working out",
        "I keep trying but nothing ever changes",
        "This is so annoying and pointless",
        "I am stuck and cannot figure out what to do",
        "Everything I try fails and it is driving me crazy",
    ],
    "nostalgic": [
        "I really miss those days, I wish I could go back",
        "I keep thinking about the past and how things used to be",
        "Those were such good times, I miss them so much",
        "I remember when everything felt simpler and better",
        "Thinking about old memories makes me feel warm and sad at the same time",
    ],
    "hopeful": [
        "I feel hopeful that things will get better",
        "I believe things are slowly improving",
        "I am not giving up, I think there is a brighter future ahead",
        "Maybe things will work out, I am trying to stay positive",
        "I feel like I am slowly moving forward",
    ],
    "calm": [
        "I feel calm and at peace right now",
        "Everything feels settled and I am relaxed",
        "I feel grounded and centered today",
        "I have let go of my worries and feel peaceful",
        "I feel okay, things are quiet and still",
    ],
    "numb": [
        "I feel completely numb and empty inside",
        "I do not feel anything at all, just going through the motions",
        "Everything feels flat and meaningless",
        "I am detached from everything, like a zombie",
        "I have stopped caring about things, nothing reaches me",
    ],
    "confused": [
        "I feel confused and do not know what is happening",
        "Nothing makes sense to me right now",
        "I am lost and cannot figure out what I think or feel",
        "Everything feels mixed up and unclear",
        "I do not understand what is going on with me",
    ],
}

# Detail labels for each emotion
EMOTION_DETAILS = {
    "happy":       "genuinely-happy",
    "sad":         "deeply-sad",
    "lonely":      "quietly-lonely",
    "anxious":     "quietly-anxious",
    "overwhelmed": "completely-overwhelmed",
    "angry":       "deeply-angry",
    "frustrated":  "quietly-frustrated",
    "nostalgic":   "warmly-nostalgic",
    "hopeful":     "cautiously-hopeful",
    "calm":        "peacefully-calm",
    "numb":        "quietly-numb",
    "confused":    "genuinely-confused",
}

_emotion_embeddings = None


def _get_emotion_embeddings():
    """Pre-compute and cache embeddings for all emotion prototypes."""
    global _emotion_embeddings
    if _emotion_embeddings is None:
        model = _get_model()
        _emotion_embeddings = {}
        for emotion, sentences in EMOTION_PROTOTYPES.items():
            vecs = model.encode(sentences)
            # Average the prototype embeddings for each emotion
            _emotion_embeddings[emotion] = np.mean(vecs, axis=0)
    return _emotion_embeddings


def detect_emotion(text: str) -> tuple[str, str, dict]:
    """
    Detects emotion from text using sentence embeddings.
    Returns (emotion, detail, scores_dict)
    """
    if not text or len(text.strip()) < 3:
        return "neutral", "quietly-present", {}

    model = _get_model()
    text_vec = model.encode([text])
    emotion_vecs = _get_emotion_embeddings()

    scores = {}
    for emotion, proto_vec in emotion_vecs.items():
        score = float(cosine_similarity(text_vec, [proto_vec])[0][0])
        scores[emotion] = round(score, 4)

    # Find best emotion
    best_emotion = max(scores, key=scores.get)
    best_score = scores[best_emotion]

    # If score is too low, return neutral
    if best_score < 0.25:
        return "neutral", "quietly-present", scores

    detail = EMOTION_DETAILS.get(best_emotion, "quietly-present")
    return best_emotion, detail, scores
