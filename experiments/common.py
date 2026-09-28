"""Shared helpers for all CLM demo experiments."""
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")  # no display needed under WSL; plots are saved as PNG
import matplotlib.pyplot as plt  # noqa: E402  (re-exported for experiments)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from clm import CLMClient, Choice, Noul, Score  # noqa: E402

# None -> CLMClient default (http://127.0.0.1:8700). Override: export CLM_URL=...
CLM_URL = os.environ.get("CLM_URL")

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Synthetic customer-support dataset (every sample has known ground truth)
# --------------------------------------------------------------------------
ISSUES = {
    "billing": [
        "my invoice was charged twice",
        "my refund still hasn't arrived after three weeks",
        "the amount on this month's invoice is wrong",
        "there is an extra fee on my bill I never agreed to",
        "my card was billed again after I cancelled my plan",
        "I was charged for the annual plan instead of monthly",
    ],
    "technical": [
        "the app crashes every time I log in",
        "your website has been down all morning",
        "the password reset link is broken",
        "the export feature returns an error 500",
        "sync stopped working after the latest update",
        "the dashboard just shows a blank page",
    ],
}

TONES = {
    "calm": ("Hi, ", " Could you take a look when you have a moment? Thanks."),
    "frustrated": ("", " This is the second time I'm writing about this and it's getting annoying."),
    "very_angry": ("", " This is RIDICULOUS!!! Nobody answers the phone and I'm about to switch to a competitor!"),
}
TONE_ORDER = list(TONES)
TONE_COLORS = {"calm": "#55A868", "frustrated": "#DD8452", "very_angry": "#C44E52"}

CONTEXT = [
    "I have been a customer for five years.",
    "I already tried again this morning.",
    "My account email is the one I'm writing from.",
    "My colleague has the same problem.",
    "I need this sorted before the end of the week.",
    "I checked the help center but found nothing useful.",
]

QUESTIONS = {
    "urgency": Noul(instructions="Is this urgent?"),
    "department": Choice(
        instructions="Which team should handle this?",
        criteria={"billing": "Charges, invoices, refunds",
                  "technical": "Bugs and outages"},
    ),
    "frustration": Score(
        instructions="How frustrated is the customer?",
        criteria=["Calm", "Frustrated", "Very angry"],
    ),
}


def make_text(rng, dept, tone):
    issue = rng.choice(ISSUES[dept])
    pre, post = TONES[tone]
    if not pre:
        issue = issue[0].upper() + issue[1:]
    filler = " ".join(rng.sample(CONTEXT, rng.randint(0, len(CONTEXT))))
    return f"Customer: {pre}{issue}." + (f" {filler}" if filler else "") + post


def make_dataset(n, seed=42, unique=True):
    """n samples mixing department x tone x length. unique=True avoids repeats
    (repeated texts can hit a cache and distort latency numbers)."""
    rng = random.Random(seed)
    rows, seen, attempts = [], set(), 0
    while len(rows) < n and attempts < n * 100:
        attempts += 1
        dept, tone = rng.choice(list(ISSUES)), rng.choice(TONE_ORDER)
        text = make_text(rng, dept, tone)
        if unique and text in seen:
            continue
        seen.add(text)
        rows.append({"id": len(rows), "true_department": dept, "true_tone": tone,
                     "state": text, "n_chars": len(text), "n_words": len(text.split())})
    return rows


# --------------------------------------------------------------------------
# Calling CLM
# --------------------------------------------------------------------------
_local = threading.local()


def get_client():
    """One CLMClient per thread (HTTP sessions aren't fully thread-safe)."""
    if not hasattr(_local, "client"):
        _local.client = CLMClient(CLM_URL) if CLM_URL else CLMClient()
    return _local.client


def to_dict(obj):
    """Turn a CLM answer object into a plain dict, whatever class it is."""
    for m in ("model_dump", "dict", "_asdict"):
        f = getattr(obj, m, None)
        if callable(f):
            try:
                return dict(f())
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"value": obj}


def flatten_answers(answers):
    """answers['department'].probabilities['billing'] -> 'department_probabilities_billing'."""
    out = {}
    for q, a in answers.items():
        for k, v in to_dict(a).items():
            if isinstance(v, dict):
                for sub, val in v.items():
                    out[f"{q}_{k}_{sub}"] = val
            elif v is None or isinstance(v, (int, float, str, bool)):
                out[f"{q}_{k}"] = v
            else:
                out[f"{q}_{k}"] = str(v)
    return out


def run_one(sample):
    row = dict(sample)
    t0 = time.perf_counter()
    try:
        r = get_client().system_one(state=sample["state"], questions=QUESTIONS)
        row["latency_ms"] = (time.perf_counter() - t0) * 1000
        row["ok"] = True
        row.update(flatten_answers(r.answers))
    except Exception as e:
        row["latency_ms"] = (time.perf_counter() - t0) * 1000
        row["ok"] = False
        row["error"] = str(e)[:300]
    return row


def warm_up(k=3):
    """Untimed calls with texts that are NOT in any dataset (so they don't pre-fill a cache)."""
    for i in range(k):
        try:
            get_client().system_one(state=f"Customer: warm-up message {i} {time.time()}",
                                    questions=QUESTIONS)
        except Exception as e:
            raise SystemExit(f"Warm-up failed - are both services running? "
                             f"Try ./scripts/03_check_services.sh\n{e}")


def run_batch(samples, workers=1, warmup=3, label=""):
    """Send all samples, return (DataFrame sorted by id, wall-clock seconds)."""
    if warmup:
        warm_up(warmup)
    rows = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, s) for s in samples]
        for k, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            if k % 20 == 0 or k == len(futures):
                print(f"  {label}{k}/{len(futures)} done")
    wall = time.perf_counter() - t0
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    failed = (~df["ok"]).sum()
    if failed:
        print(f"  WARNING: {failed} request(s) failed. First error: {df.loc[~df['ok'], 'error'].iloc[0]}")
    return df, wall


def latency_stats(s):
    s = pd.Series(s).dropna()
    return {"mean": s.mean(), "p50": s.quantile(.5), "p90": s.quantile(.9),
            "p95": s.quantile(.95), "p99": s.quantile(.99), "max": s.max()}


def print_stats(name, s):
    st = latency_stats(s)
    print(f"{name:>12}: " + " | ".join(f"{k} {v:.1f}" for k, v in st.items()) + "  (ms)")


def out(name):
    return os.path.join(RESULTS_DIR, name)


def save_fig(fig, name):
    path = out(name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved plot -> {path}")
