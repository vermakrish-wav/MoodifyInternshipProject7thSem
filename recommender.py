"""Mood-based content recommender: mood -> target audio-feature profile -> ranked tracks."""
import re
import numpy as np, pandas as pd

FEATURES = ["valence", "energy", "danceability", "acousticness", "tempo"]
# target profile (tempo in BPM) and feature weights per mood
MOODS = {
    "happy":     dict(target=[0.85, 0.75, 0.78, 0.25, 125], w=[3, 2, 2, 1, 1]),
    "sad":       dict(target=[0.15, 0.25, 0.38, 0.75,  78], w=[3, 2, 1, 2, 1]),
    "calm":      dict(target=[0.50, 0.15, 0.42, 0.88,  88], w=[1, 3, 1, 3, 1]),
    "energetic": dict(target=[0.68, 0.92, 0.72, 0.08, 142], w=[1, 3, 2, 1, 2]),
    "angry":     dict(target=[0.22, 0.90, 0.45, 0.08, 152], w=[3, 3, 1, 1, 2]),
    "romantic": dict(target=[0.60, 0.35, 0.55, 0.55, 95], w=[2, 2, 1, 2, 1]),
    "focused":  dict(target=[0.45, 0.30, 0.35, 0.60, 100], w=[1, 3, 2, 2, 1]),
    "nostalgic": dict(target=[0.40 , 0.40, 0.20, 0.80, 90], w=[3, 2, 1, 3, 3]),
    "heartbroken": dict(target=[0.15, 0.40, 0.50, 0.90, 100], w=[3, 2, 1, 3, 2]),
    "confident": dict(target=[0.80, 0.97, 0.80, 0.20, 100],w=[3, 3, 2, 1, 2]),
    "anxious": dict(target=[0.45, 0.50, 0.30, 0.87, 85], w=[2, 2, 1, 3, 2]),
    "dreamy": dict(target=[0.75, 0.70, 0.50, 0.40, 70], w=[2, 2, 1, 2, 3])
}

LEXICON = {
    "happy": "happy joy joyful cheerful great excited glad smile fun party good celebrate".split(),
    "sad": "sad cry crying lonely heartbroken down depressed blue miss upset tears gloomy".split(),
    "calm": "calm relax relaxed peaceful sleepy chill study tired quiet rest serene".split(),
    "energetic": "energetic workout gym run pumped hyped active motivated energy dance".split(),
    "angry": "angry mad furious annoyed rage frustrated irritated hate livid".split(),
    "romantic": "love loved romantic crush date valentine affection tender adore sweetheart intimate passion".split(),
    "focused": "focus focused study studying work concentrate coding exam semester exams productive assignment deadline".split(),
    "nostalgic": "nostalgic memories remember childhood throwback old days reminisce past familiar flashback school".split(),
    "heartbroken": "heartbroken breakup ex missing missyou abandoned betrayal separation unrequited lostlove rejection".split(),
    "confident": "confident ambition determined fearless winning success powerful unstoppable champion bold dominant".split(),
    "anxious": "anxious anxiety stressed stress worried nervous panic overwhelmed tense restless pressure deadline".split(),
    "dreamy": "dreamy vibing laidback atmospheric floating hazy smooth ethereal ambient night surreal".split()
}

_LO = np.array([0, 0, 0, 0, 50.0]); _HI = np.array([1, 1, 1, 1, 200.0])


def detect_mood(text):
    """Keyword lexicon classifier. Returns (mood, hits) or (None, {}) when nothing matches."""
    words = re.findall(r"[a-z']+", text.lower())
    hits = {m: sum(w in ks for w in words) for m, ks in LEXICON.items()}
    best = max(hits, key=hits.get)
    return (best, hits) if hits[best] > 0 else (None, hits)


def _norm(v):
    return (np.asarray(v, float) - _LO) / (_HI - _LO)


def _score(df, target, weights):
    """Similarity in [0, 1] between every track and a target profile (raw units, e.g. tempo in BPM)."""
    X = _norm(df[FEATURES].to_numpy(float))
    t = _norm(target)
    w = np.asarray(weights, float); w = w / w.sum()
    dist = np.sqrt((w * (X - t) ** 2).sum(axis=1))
    return 1 - dist / np.sqrt(w.sum())            # similarity in [0, 1]


def score_tracks(df, mood):
    cfg = MOODS[mood]
    return _score(df, cfg["target"], cfg["w"])


def _why(row, target, weights, n=2):
    order = np.argsort(-np.asarray(weights, float), kind="stable")[:n]
    parts = []
    for i in order:
        f = FEATURES[i]; v = float(row[f]); t = float(target[i])
        fmt = (lambda x: f"{x:.0f}") if f == "tempo" else (lambda x: f"{x:.2f}")
        parts.append(f"{f} {fmt(v)} (target {fmt(t)})")
    return " · ".join(parts)


def _pick(d, k, max_per_artist, chosen=None):
    out, count = [], {}
    for _, r in d.iterrows():
        if count.get(r.artist, 0) >= max_per_artist:
            continue
        count[r.artist] = count.get(r.artist, 0) + 1
        out.append(r)
        if len(out) == k:
            break
    return out


def recommend(df, mood, k=10, max_per_artist=2, energy_bias=0.0):
    """Top-k tracks for a mood with an artist-diversity cap. energy_bias in [-1, 1] nudges energy."""
    if mood not in MOODS:
        raise ValueError(f"unknown mood: {mood}")
    if k < 1:
        raise ValueError("k must be >= 1")
    d = df.copy()
    d["score"] = score_tracks(d, mood) + 0.1 * energy_bias * (d["energy"] - 0.5)
    d = d.sort_values("score", ascending=False)
    out = pd.DataFrame(_pick(d, k, max_per_artist)).reset_index(drop=True)
    cfg = MOODS[mood]
    out["why"] = [_why(r, cfg["target"], cfg["w"]) for _, r in out.iterrows()]
    return out


def journey(df, start, goal, k=10, max_per_artist=2):
    """Mood Journey: a playlist that moves gradually from the `start` mood to the `goal` mood.

    Step i of k blends the two mood profiles (target and weights) with alpha = i / (k - 1), so step 1 is
    pure `start`, the last step is pure `goal`, and the tracks in between are stepping stones. Each
    track is used once and the artist cap applies across the whole journey.
    """
    for m in (start, goal):
        if m not in MOODS:
            raise ValueError(f"unknown mood: {m}")
    if k < 1:
        raise ValueError("k must be >= 1")
    a, b = MOODS[start], MOODS[goal]
    used, count, rows = set(), {}, []
    for i in range(k):
        alpha = i / (k - 1) if k > 1 else 0.0
        target = (1 - alpha) * np.array(a["target"], float) + alpha * np.array(b["target"], float)
        w = (1 - alpha) * np.array(a["w"], float) + alpha * np.array(b["w"], float)
        d = df.copy(); d["score"] = _score(d, target, w)
        for _, r in d.sort_values("score", ascending=False).iterrows():
            if r.track_id in used or count.get(r.artist, 0) >= max_per_artist:
                continue
            used.add(r.track_id); count[r.artist] = count.get(r.artist, 0) + 1
            r = r.copy(); r["step"] = i + 1; r["blend"] = round(alpha, 3); r["why"] = _why(r, target, w)
            rows.append(r); break
    return pd.DataFrame(rows).reset_index(drop=True)