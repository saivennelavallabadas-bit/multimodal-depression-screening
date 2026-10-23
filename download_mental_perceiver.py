
from pathlib import Path
import urllib.request

base_url = (
    "https://shuyeit.oss-cn-guangzhou.aliyuncs.com/"
    "Mmpsy_OpenData/data/"
)

data_dir = Path("data/mmpsy-data-main/data")
data_dir.mkdir(parents=True, exist_ok=True)

files = [
    f"np_data.zip.{i:03d}" for i in range(1, 6)
]

for filename in files:
    destination = data_dir / filename

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already exists: {filename}")
        continue

    temporary = destination.with_name(filename + ".part")
    url = base_url + filename

    print(f"\nDownloading {filename}...")

    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            total = 0

            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    total += len(chunk)

        if total == 0:
            raise RuntimeError("Downloaded file is empty")

        temporary.replace(destination)
        print(f"Completed: {filename} ({total / 1024**2:.1f} MB)")

    except Exception as error:
        temporary.unlink(missing_ok=True)
        print(f"FAILED: {filename}: {error}")
        print("Fix the issue and run this script again.")
        raise

print("\nAll five archive parts are downloaded.")
