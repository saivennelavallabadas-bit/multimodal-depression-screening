
from pathlib import Path

folder = Path("label")
files = list(folder.glob("*.csv"))

print("Total label files:", len(files))
print("Depression files:", sum("depression" in f.name.lower() for f in files))
print("Normal files:", sum("normal" in f.name.lower() for f in files))

print("\nLast 20 filenames:")
for f in sorted(files)[-20:]:
    print(f.name)
