"""Experiment 4 - Concurrency: how much throughput do parallel clients buy, and at what latency cost?"""
import argparse

import pandas as pd

from common import latency_stats, make_dataset, out, plt, run_batch, save_fig

ap = argparse.ArgumentParser()
ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--n", type=int, default=60, help="requests per level")
args = ap.parse_args()

summary = []
for i, w in enumerate(args.levels):
    # fresh texts per level so no level benefits from another level's cache
    data = make_dataset(args.n, seed=1000 + i)
    df, wall = run_batch(data, workers=w, warmup=2 if i == 0 else 0, label=f"[workers={w}] ")
    ok = df[df["ok"]]
    st = latency_stats(ok["latency_ms"])
    summary.append({"workers": w, "throughput_rps": len(ok) / wall, **st})
    print(f"workers={w}: {len(ok) / wall:.2f} req/s, p50 {st['p50']:.1f} ms, p95 {st['p95']:.1f} ms")

s = pd.DataFrame(summary)
s.to_csv(out("exp04_concurrency.csv"), index=False)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(s["workers"], s["throughput_rps"], "o-")
ax[0].set(title="Throughput vs concurrent clients", xlabel="workers", ylabel="requests / s")
ax[1].plot(s["workers"], s["p50"], "o-", label="p50")
ax[1].plot(s["workers"], s["p95"], "s--", label="p95")
ax[1].set(title="Latency vs concurrent clients", xlabel="workers", ylabel="ms")
ax[1].legend()
for a in ax:
    a.set_xticks(s["workers"])
fig.suptitle("Exp 4 - Concurrency sweep", fontsize=14)
save_fig(fig, "exp04_concurrency.png")
