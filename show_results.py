
import json
from pathlib import Path

path = Path("artifacts/lmvd_baseline/metrics.json")

with path.open("r", encoding="utf-8") as f:
    results = json.load(f)

print("Dataset note:")
print(results.get("warning", "No warning field found."))

print("\nModel comparison:")
for name, metrics in results["models"].items():
    print(f"\n{name.upper()}")
    print("Validation:", metrics["validation"])
    print("Test:", metrics["test"])
