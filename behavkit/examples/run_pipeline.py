"""
Minimal example: with config.yaml filled in, the entire pipeline runs from
a single function call. All results are written as CSV/JSON files (paths
defined in config.yaml) -- open them in whatever stats/plotting tool you
prefer.

    1. Copy config.example.yaml to config.yaml
    2. Fill in your paths, bodyparts, and behavior labels
    3. Run this script
"""
import behavkit as bk

results = bk.run_pipeline("config.yaml")

print(results.keys())

if "modeling" in results:
    print(results["modeling"]["report_text"])

if "sequences" in results:
    print(results["sequences"]["sequences"].head())

if "threshold_detection" in results:
    print(f"Selected threshold: {results['threshold_detection']['threshold']:.4f}")
    print(results["threshold_detection"]["bout_metrics"].head())
    