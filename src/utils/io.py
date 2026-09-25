import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List
import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_json(path: str) -> Dict[str, Any]:
    """Loads a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_run_directory(base_dir: str = "data/results") -> str:
    """Creates a timestamped run directory and returns its path."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_predictions_jsonl(predictions: List[Dict[str, Any]], filepath: str) -> None:
    """Saves predictions list into a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

def save_predictions_csv(predictions: List[Dict[str, Any]], filepath: str) -> None:
    """Saves predictions list into a CSV file."""
    if not predictions:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = list(predictions[0].keys())
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pred in predictions:
            row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in pred.items()}
            writer.writerow(row)

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Saves data to a JSON file formatted with indentations."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
