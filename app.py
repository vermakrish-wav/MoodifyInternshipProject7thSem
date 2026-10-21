from urllib.parse import quote
import pandas as pd
from flask import Flask, render_template, request, jsonify
from recommender import MOODS, recommend, journey, detect_mood

app = Flask(__name__)
DF = pd.read_csv("data/csvmoodfiy.csv")
EMOJI = {"happy": "😊", "sad": "😢", "calm": "😌", "energetic": "⚡", "angry": "😠", "romantic":"💕","focused":"🎯","nostalgic":"☹️","heartbroken":"💔","confident":"😁", "anxious":"🫪","dreamy":"💭"}


DEFAULT_K, MAX_K = 10, 500


def get_k(value):
    """Parse the requested number of tracks: whole number between 1 and MAX_K, otherwise the default."""
    try:
        return max(1, min(MAX_K, int(value)))
    except (TypeError, ValueError):
        return DEFAULT_K


def spotify_url(r):
    """Exact track link when the dataset has a real 22-character Spotify ID, otherwise a search link."""
    tid = str(r.get("track_id", ""))
    if len(tid) == 22 and tid.isalnum():
        return "https://open.spotify.com/track/" + tid
    return "https://open.spotify.com/search/" + quote(f"{r['title']} {r['artist']}")


@app.route("/")
def index():
    mood = request.args.get("mood", "")
    text = request.args.get("text", "").strip()
    k = get_k(request.args.get("k"))
    start, goal = request.args.get("start", ""), request.args.get("goal", "")
    msg, tracks, trip = "", [], False
    if start in MOODS and goal in MOODS:
        res = journey(DF, start, goal, k=k)
        tracks = [dict(r, url=spotify_url(r)) for r in res.to_dict("records")]
        trip = True
    elif text and not mood:
        mood, _ = detect_mood(text)
        mood = mood or ""
        msg = f'Detected mood: {mood}' if mood else "Could not detect a mood. Pick one below."
    if mood in MOODS and not trip:
        res = recommend(DF, mood, k=k, energy_bias=float(request.args.get("energy", 0)))
        tracks = [dict(r, url=spotify_url(r)) for r in res.to_dict("records")]
    return render_template("index.html", moods=MOODS, emoji=EMOJI, mood=mood, tracks=tracks, text=text, msg=msg,
                           trip=trip, start=start, goal=goal, k=k, max_k=MAX_K)


@app.route("/api/recommend/<mood>")
def api(mood):
    if mood not in MOODS:
        return jsonify(error="unknown mood"), 404
    return jsonify(recommend(DF, mood, k=get_k(request.args.get("k"))).to_dict("records"))


@app.route("/api/journey/<start>/<goal>")
def api_journey(start, goal):
    if start not in MOODS or goal not in MOODS:
        return jsonify(error="unknown mood"), 404
    return jsonify(journey(DF, start, goal, k=get_k(request.args.get("k"))).to_dict("records"))


if __name__ == "__main__":
    app.run(debug=False, port=5000)