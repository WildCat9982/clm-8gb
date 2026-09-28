"""Experiment 5 - Caching: are repeated inputs answered from a cache?

In the first benchmark a handful of requests took ~1-2 ms while the rest took 40-80 ms.
Hypothesis: those were duplicate texts served from a cache. This experiment tests it.
"""
import argparse

import pandas as pd

from common import make_dataset, out, plt, print_stats, run_batch, save_fig

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40)
args = ap.parse_args()

unique = make_dataset(args.n, seed=5001)
one = make_dataset(1, seed=5002)[0]
repeated = [{**one, "id": i} for i in range(args.n)]

print("Sending the SAME text repeatedly...")
df_rep, _ = run_batch(repeated, workers=1, warmup=3)
print("Sending UNIQUE texts...")
df_uni, _ = run_batch(unique, workers=1, warmup=0)

df_rep["mode"], df_uni["mode"] = "repeated", "unique"
pd.concat([df_rep, df_uni]).to_csv(out("exp05_cache.csv"), index=False)

print()
print_stats("repeated", df_rep["latency_ms"])
print_stats("unique", df_uni["latency_ms"])
ratio = df_uni["latency_ms"].median() / max(df_rep["latency_ms"].iloc[1:].median(), 1e-6)
print(f"\nUnique / repeated median ratio: {ratio:.1f}x "
      f"-> {'caching is very likely' if ratio > 5 else 'no strong cache effect'}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].boxplot([df_rep["latency_ms"], df_uni["latency_ms"]])
ax[0].set_xticks([1, 2], ["repeated text", "unique texts"])
ax[0].set(title="Latency: repeated vs unique input", ylabel="ms")
ax[1].plot(df_rep["id"], df_rep["latency_ms"], "o-", ms=4, label="repeated")
ax[1].plot(df_uni["id"], df_uni["latency_ms"], "o-", ms=4, alpha=.7, label="unique")
ax[1].set(title="Per-request latency (note the first repeated call)", xlabel="request #", ylabel="ms")
ax[1].legend()
fig.suptitle("Exp 5 - Cache effect", fontsize=14)
save_fig(fig, "exp05_cache.png")
