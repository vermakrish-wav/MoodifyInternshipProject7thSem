import pandas as pd, json
from recommender import recommend, MOODS
df = pd.read_csv("data/tracks.csv"); res = {}
for m in MOODS:
    r = recommend(df, m, k=10)
    res[m] = dict(p10=float((r.true_mood == m).mean()), valence=float(r.valence.mean()), energy=float(r.energy.mean()),
                  artists=int(r.artist.nunique()), genres=int(r.genre.nunique()), top=float(r.score.iloc[0]), last=float(r.score.iloc[-1]))
# baseline: random 10
import numpy as np
rng = np.random.default_rng(0)
base = np.mean([(df.sample(10, random_state=int(s)).true_mood == m).mean() for m in MOODS for s in range(200)])
res["random_baseline_p10"] = float(base)
json.dump(res, open("eval.json", "w"), indent=1); print(json.dumps(res, indent=1))
