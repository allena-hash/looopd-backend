import math
import re
from collections import Counter

# ── Lightweight TF-IDF cosine similarity ─────────────────────
# No ML model needed — pure Python, zero memory overhead

STOP = set(['i','a','the','is','it','in','of','and','to','was','my','me',
'you','that','this','we','do','not','so','but','just','for','on','at','be',
'have','with','they','he','she','are','were','an','im','its','dont','cant',
'get','go','very','much','also','even','still','really','like','know','feel',
'think','about','what','when','how','why','who','there','here','your','our',
'its','been','has','had','will','would','could','should','may','might','shall'])

def tokenize(text):
    words = re.findall(r"[a-z']+", text.lower())
    return [w for w in words if w not in STOP and len(w) > 2]

def tfidf_vector(tokens, all_docs_tokens):
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    vec = {}
    N = len(all_docs_tokens)
    for word, count in tf.items():
        df = sum(1 for doc in all_docs_tokens if word in doc)
        idf = math.log((N + 1) / (df + 1)) + 1
        vec[word] = (count / total) * idf
    return vec

def cosine(v1, v2):
    keys = set(v1) & set(v2)
    if not keys:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in keys)
    mag1 = math.sqrt(sum(x*x for x in v1.values()))
    mag2 = math.sqrt(sum(x*x for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

def top_n_similar(query, candidates, n=2):
    if not candidates:
        return [], []
    valid = [(i, c) for i, c in enumerate(candidates) if c and c.strip()]
    if not valid:
        return [], []
    indices, texts = zip(*valid)
    all_tokens = [set(tokenize(t)) for t in texts]
    query_tokens = tokenize(query)
    all_docs = [query_tokens] + [tokenize(t) for t in texts]
    query_vec = tfidf_vector(query_tokens, [set(d) for d in all_docs])
    scores = []
    for text in texts:
        t = tokenize(text)
        vec = tfidf_vector(t, [set(d) for d in all_docs])
        scores.append(cosine(query_vec, vec))
    n = min(n, len(scores))
    import heapq
    top = heapq.nlargest(n, range(len(scores)), key=lambda i: scores[i])
    return [indices[i] for i in top], [scores[i] for i in top]

def most_similar(query, candidates):
    indices, scores = top_n_similar(query, candidates, n=1)
    if not indices:
        return 0, 0.0
    return indices[0], scores[0]


# ── Emotion detection using weighted phrases ─────────────────
EMOTIONS = {
    'happy': {
        'detail': 'genuinely-happy',
        'phrases': [
            ('so happy',3),('really happy',3),('feeling great',3),('so good',3),
            ('having fun',3),('so much fun',3),('love it',3),('amazing',2),
            ('wonderful',2),('fantastic',2),('great',2),('awesome',2),('joy',2),
            ('enjoying',2),('smile',2),('laughing',2),('grateful',2),('blessed',2),
            ('proud',2),('excited',2),('thrilled',2),('good',1),('nice',1),('glad',1),
        ]
    },
    'sad': {
        'detail': 'deeply-sad',
        'phrases': [
            ('so sad',3),('really sad',3),('feel terrible',3),('been crying',3),
            ('heartbroken',3),('devastated',3),('feel awful',3),('feeling low',3),
            ('sad',2),('cry',2),('crying',2),('tears',2),('depressed',2),('hurt',2),
            ('pain',2),('broken',2),('hopeless',2),('helpless',2),('miserable',2),
            ('unhappy',2),('grief',2),('lost',2),('miss',2),('down',1),('low',1),
        ]
    },
    'lonely': {
        'detail': 'quietly-lonely',
        'phrases': [
            ('feel so alone',3),('no one cares',3),('nobody understands',3),
            ('feel invisible',3),('no one to talk',3),('completely alone',3),
            ('alone',2),('lonely',2),('nobody',2),('no one',2),('isolated',2),
            ('left out',2),('excluded',2),('forgotten',2),('ignored',2),
            ('abandoned',2),('disconnected',2),('no friends',2),('by myself',2),
        ]
    },
    'anxious': {
        'detail': 'quietly-anxious',
        'phrases': [
            ('so anxious',3),('really worried',3),('cannot stop worrying',3),
            ('heart racing',3),('panic attack',3),('mind racing',3),
            ('anxious',2),('anxiety',2),('worried',2),('worry',2),('scared',2),
            ('nervous',2),('panic',2),('overthinking',2),('afraid',2),('fear',2),
            ('stressed',2),('stress',2),('dread',2),('tense',2),('uneasy',2),
        ]
    },
    'overwhelmed': {
        'detail': 'completely-overwhelmed',
        'phrases': [
            ('too much to handle',3),('cannot cope',3),('completely overwhelmed',3),
            ('breaking point',3),('drowning in',3),('at my limit',3),
            ('overwhelmed',2),('too much',2),('exhausted',2),('drained',2),
            ('burnout',2),('burnt out',2),('no energy',2),('swamped',2),
            ('buried',2),('losing control',2),('out of control',2),
        ]
    },
    'angry': {
        'detail': 'deeply-angry',
        'phrases': [
            ('so angry',3),('really angry',3),('furious',3),('absolutely livid',3),
            ('angry',2),('anger',2),('mad',2),('rage',2),('hate',2),
            ('outraged',2),('fed up',2),('betrayed',2),('unfair',2),('livid',2),
        ]
    },
    'frustrated': {
        'detail': 'quietly-frustrated',
        'phrases': [
            ('so frustrated',3),('nothing is working',3),('tried everything',3),
            ('frustrated',2),('annoyed',2),('irritated',2),('stuck',2),
            ('nothing works',2),('pointless',2),('useless',2),('going nowhere',2),
        ]
    },
    'nostalgic': {
        'detail': 'warmly-nostalgic',
        'phrases': [
            ('wish i could go back',3),('those were the days',3),('really miss those',3),
            ('remember when',2),('used to',2),('back then',2),('those days',2),
            ('childhood',2),('years ago',2),('miss the old',2),('simpler times',2),
        ]
    },
    'hopeful': {
        'detail': 'cautiously-hopeful',
        'phrases': [
            ('things will get better',3),('slowly getting there',3),('not giving up',3),
            ('hope',2),('hopeful',2),('believe',2),('getting better',2),
            ('improving',2),('moving forward',2),('keep going',2),('one day',1),
        ]
    },
    'calm': {
        'detail': 'peacefully-calm',
        'phrases': [
            ('feeling really calm',3),('at peace',3),('totally relaxed',3),
            ('calm',2),('peaceful',2),('relaxed',2),('settled',2),('grounded',2),
            ('okay now',2),('feeling better now',2),('letting go',2),('still',1),
        ]
    },
    'numb': {
        'detail': 'quietly-numb',
        'phrases': [
            ('feel completely numb',3),('feel nothing',3),('going through motions',3),
            ('just existing',3),('disconnected from everything',3),
            ('numb',2),('empty inside',2),('hollow',2),('detached',2),
            ('dont care anymore',2),('stopped caring',2),('like a zombie',2),
        ]
    },
    'confused': {
        'detail': 'genuinely-confused',
        'phrases': [
            ('makes no sense',3),('do not understand',3),('completely lost',3),
            ('confused',2),('mixed up',2),('cannot figure',2),('unclear',2),
            ('uncertain',2),('not sure what',2),('lost track',2),
        ]
    },
}

def detect_emotion(text):
    if not text or len(text.strip()) < 3:
        return 'neutral', 'quietly-present', {}
    tl = text.lower()
    scores = {}
    for emo, data in EMOTIONS.items():
        score = 0
        for phrase, weight in data['phrases']:
            if phrase in tl:
                score += weight
        if score > 0:
            scores[emo] = score
    if not scores:
        return 'neutral', 'quietly-present', {}
    best = max(scores, key=scores.get)
    if scores[best] < 2:
        return 'neutral', 'quietly-present', scores
    detail = EMOTIONS[best]['detail']
    return best, detail, scores
