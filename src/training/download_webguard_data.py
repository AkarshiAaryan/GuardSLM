import json
import os
import urllib.request
from typing import Dict, Any

WEBGARD_REPO_URL = "https://raw.githubusercontent.com/OSU-NLP-Group/WebGuard/main"
FILES_TO_DOWNLOAD = {
    "train_a11y": "data/monitor_factory_train_a11y.json",
    "test_action": "data/monitor_factory_test_action.json",
    "test_domain": "data/monitor_factory_test_domain.json",
    "test_website": "data/monitor_factory_test_website.json",
}

def download_file(rel_path: str, dest_dir: str = "data/webguard") -> str:
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(rel_path)
    dest_path = os.path.join(dest_dir, filename)
    url = f"{WEBGARD_REPO_URL}/{rel_path}"
    print(f"Downloading {url} -> {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "GuardSLM/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        out_file.write(resp.read())
    print(f"  Successfully saved {filename} ({os.path.getsize(dest_path):,} bytes)")
    return dest_path

def download_all_webguard_datasets(dest_dir: str = "data/webguard") -> Dict[str, str]:
    paths = {}
    for key, rel_path in FILES_TO_DOWNLOAD.items():
        try:
            paths[key] = download_file(rel_path, dest_dir)
        except Exception as e:
            print(f"  Warning: Failed to download {rel_path}: {e}")
    
    # Inspect train statistics
    train_path = paths.get("train_a11y")
    if train_path and os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        labels = {}
        for sample in train_data:
            messages = sample.get("messages", [])
            for m in messages:
                if m.get("role") == "assistant":
                    content = m.get("content", "")
                    try:
                        parsed = json.loads(content)
                        lvl = parsed.get("risky_level", "UNKNOWN")
                    except Exception:
                        lvl = "UNPARSEABLE"
                    labels[lvl] = labels.get(lvl, 0) + 1
        print("\n[WebGuard Dataset Summary]")
        print(f"Total Accessibility Training Samples: {len(train_data)}")
        print(f"Label Distribution: {labels}")
    return paths

if __name__ == "__main__":
    download_all_webguard_datasets()
