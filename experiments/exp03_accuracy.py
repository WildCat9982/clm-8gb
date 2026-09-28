"""Experiment 3 - Quality: are the answers right, and is the confidence honest?"""
import argparse

import numpy as np
import pandas as pd

from common import ISSUES, TONE_ORDER, make_dataset, out, plt, run_batch, save_fig

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--workers", type=int, default=2)
args = ap.parse_args()

df, _ = run_batch(make_dataset(args.n, args.seed), workers=args.workers)
df.to_csv(out("exp03_accuracy.csv"), index=False)
ok = df[df["ok"]]


def heatmap(a, table, title, xlabel, ylabel):
    a.imshow(table.values, cmap="Blues")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            a.text(j, i, table.values[i, j], ha="center", va="center", fontsize=12)
    a.set_xticks(range(table.shape[1]), table.columns, rotation=20)
    a.set_yticks(range(table.shape[0]), table.index)
    a.set(title=title, xlabel=xlabel, ylabel=ylabel)


def answer_vs_tone(a, prefix):
    """Plot a question's answer against the true tone, whatever shape the answer has."""
    prob = [c for c in ok.columns if c.startswith(f"{prefix}_probabilities_")]
    other = [c for c in ok.columns if c.startswith(prefix + "_") and c not in prob]
    if prob:  # mean probability of each option, grouped by true tone
        means = ok.groupby("true_tone")[prob].mean().reindex(TONE_ORDER)
        means.columns = [c.replace(f"{prefix}_probabilities_", "") for c in prob]
        means.plot(kind="bar", ax=a, rot=0)
        a.set(title=f"{prefix}: mean probability by true tone", ylabel="probability", xlabel="")
        return
    num = [c for c in other if pd.api.types.is_numeric_dtype(ok[c]) and not pd.api.types.is_bool_dtype(ok[c])]
    if num:
        a.boxplot([ok[ok["true_tone"] == t][num[0]].dropna() for t in TONE_ORDER])
        a.set_xticks(range(1, len(TONE_ORDER) + 1), TONE_ORDER)
        a.set(title=f"{prefix}: {num[0]} by true tone")
        return
    if other:
        t = pd.crosstab(ok["true_tone"], ok[other[0]].astype(str)).reindex(TONE_ORDER, fill_value=0)
        heatmap(a, t, f"{prefix} vs true tone", other[0], "true tone")
        return
    a.text(.5, .5, f"no '{prefix}_*' columns found", ha="center"); a.axis("off")


fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# Department confusion matrix
labels = list(ISSUES)
acc = float("nan")
if "department_choice" in ok:
    correct = ok["department_choice"] == ok["true_department"]
    acc = correct.mean()
    cm = pd.crosstab(ok["true_department"], ok["department_choice"]).reindex(
        index=labels, columns=labels, fill_value=0)
    heatmap(ax[0, 0], cm, f"Department confusion (accuracy {acc:.1%})", "predicted", "true")

    # Confidence: correct vs wrong
    prob = [c for c in ok.columns if c.startswith("department_probabilities_")]
    conf = ok[prob].max(axis=1)
    bins = np.linspace(.5, 1, 21)
    a = ax[0, 1]
    a.hist(conf[correct], bins=bins, alpha=.8, color="#55A868", label="correct")
    a.hist(conf[~correct], bins=bins, alpha=.8, color="#C44E52", label="wrong")
    a.set(title="Department confidence (max probability)", xlabel="probability", ylabel="requests")
    a.legend()
    print(f"\nDepartment accuracy: {acc:.1%}")
    print("Per true department:")
    print(correct.groupby(ok["true_department"]).mean().map("{:.1%}".format).to_string())

answer_vs_tone(ax[1, 0], "frustration")
answer_vs_tone(ax[1, 1], "urgency")

fig.suptitle(f"Exp 3 - Quality on {len(ok)} labelled samples", fontsize=14)
save_fig(fig, "exp03_accuracy.png")
print("Tip: open results/exp03_accuracy.csv to see every raw answer field.")
