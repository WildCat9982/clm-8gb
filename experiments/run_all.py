"""Run experiments 2-5 in order (exp 1 is best run interactively)."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

for name in ["exp02_latency", "exp03_accuracy", "exp04_concurrency", "exp05_cache"]:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    sys.argv = [name]
    runpy.run_path(os.path.join(HERE, f"{name}.py"), run_name="__main__")
print("\nAll done. Plots and CSVs are in results/")
