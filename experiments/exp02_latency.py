"""Experiment 2 - Latency profile: how fast is CLM, and what drives latency?"""
import argparse

import numpy as np

from common import (TONE_COLORS, TONE_ORDER, make_dataset, out, plt,
                    print_stats, run_batch, save_fig)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

df, wall = run_batch(make_dataset(args.n, args.seed), workers=1)
df.to_csv(out("exp02_latency.csv"), index=False)
ok = df[df["ok"]]
lat = ok["latency_ms"]

print(f"\n{len(ok)}/{len(df)} ok, {len(ok) / wall:.2f} req/s (1 worker)")
print_stats("end-to-end", lat)

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
p50, p95 = lat.quantile(.5), lat.quantile(.95)

a = ax[0, 0]
a.hist(lat, bins=30, color="#4C72B0", alpha=.85)
a.axvline(p50, ls="--", c="k", label=f"p50 {p50:.0f} ms")
a.axvline(p95, ls=":", c="r", label=f"p95 {p95:.0f} ms")
a.set(title="Latency distribution", xlabel="ms", ylabel="requests"); a.legend()

a = ax[0, 1]
a.plot(ok["id"], lat, ".", alpha=.5, label="request")
a.plot(ok["id"], lat.rolling(10, min_periods=1).median(), c="k", label="rolling median (10)")
a.set(title="Latency over the run", xlabel="request id", ylabel="ms"); a.legend()

a = ax[1, 0]
for t in TONE_ORDER:
    s = ok[ok["true_tone"] == t]
    a.scatter(s["n_words"], s["latency_ms"], s=18, alpha=.7, c=TONE_COLORS[t], label=t)
m, b = np.polyfit(ok["n_words"], lat, 1)
xs = np.linspace(ok["n_words"].min(), ok["n_words"].max(), 50)
a.plot(xs, m * xs + b, "k--", label=f"fit {m:.2f} ms/word")
a.set(title="Latency vs input length", xlabel="words", ylabel="ms"); a.legend()

a = ax[1, 1]
a.boxplot([ok[ok["true_tone"] == t]["latency_ms"] for t in TONE_ORDER])
a.set_xticks(range(1, len(TONE_ORDER) + 1), TONE_ORDER)
a.set(title="Latency by customer tone", ylabel="ms")

fig.suptitle(f"Exp 2 - Latency: {len(ok) / wall:.2f} req/s, p50 {p50:.0f} ms, p95 {p95:.0f} ms", fontsize=14)
save_fig(fig, "exp02_latency.png")
