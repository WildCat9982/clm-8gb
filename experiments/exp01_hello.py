"""Experiment 1 - Hello CLM: one request, see every field CLM returns."""
import time

from common import QUESTIONS, get_client, to_dict

STATE = "Customer: my invoice was charged twice and nobody answers the phone!"

client = get_client()
for attempt in (1, 2):
    t0 = time.perf_counter()
    r = client.system_one(state=STATE, questions=QUESTIONS)
    print(f"Call {attempt}: {(time.perf_counter() - t0) * 1000:.1f} ms")

print(f"\nInput: {STATE}")
for q, a in r.answers.items():
    print(f"\n[{q}]  ({type(a).__name__})")
    for k, v in to_dict(a).items():
        print(f"   {k}: {v}")

print("\nQuestions to discuss:")
print(" - Why is call 2 faster than call 1?")
print(" - Do the probabilities look calibrated, or overconfident?")
