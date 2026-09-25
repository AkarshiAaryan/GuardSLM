#!/usr/bin/env python3
import argparse
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.validator import validate_dataset_file
from src.data.loader import load_test_cases
from src.utils.logging import setup_logger

logger = setup_logger("validate_cases")

def main():
    parser = argparse.ArgumentParser(description="Validate test case JSON dataset against schema.")
    parser.add_argument("--cases", type=str, default="data/dummy/dummy_cases.json", help="Path to dataset JSON file.")
    parser.add_argument("--schema", type=str, default="schemas/test_case_schema.json", help="Path to JSON schema file.")
    args = parser.parse_args()

    logger.info(f"Validating dataset file: {args.cases}")
    is_valid, errors = validate_dataset_file(args.cases, args.schema)

    if is_valid:
        cases = load_test_cases(args.cases)
        logger.info(f"SUCCESS: Dataset '{args.cases}' is valid and contains {len(cases)} test cases!")
        sys.exit(0)
    else:
        logger.error(f"FAILURE: Dataset '{args.cases}' failed validation with {len(errors)} errors:")
        for err in errors:
            logger.error(f"  - {err}")
        sys.exit(1)

if __name__ == "__main__":
    main()
