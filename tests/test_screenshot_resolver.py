"""Test screenshot resolver, PDF alias mapping, and missing screenshot handling."""

import os
from src.benchmark.data.loader import BenchmarkLoader
from src.benchmark.vision.screenshot_resolver import ScreenshotResolver


def test_screenshot_resolution_existing_case():
    loader = BenchmarkLoader()
    cases = loader.get_model_inputs(limit=5)
    resolver = ScreenshotResolver()

    # Case 0 is GSLM-P001-A (PDF6: 1000114630.pdf, page 9)
    scr_path, meta = resolver.resolve_case_screenshot(cases[0])
    assert meta["status"] == "resolved"
    assert scr_path is not None
    assert os.path.isfile(scr_path)
    assert os.path.getsize(scr_path) > 0
    assert meta["page_number"] == 9
    assert meta["pdf_id"] == "PDF6"


def test_alias_resolution_for_pdf1():
    resolver = ScreenshotResolver()
    # The manifest references 1000114602(1).pdf, local file is 1000114602.pdf
    pdf_path = resolver.find_pdf_file("1000114602(1).pdf")
    assert pdf_path is not None
    assert os.path.isfile(pdf_path)


def test_missing_pdf_handling():
    resolver = ScreenshotResolver(search_paths=["/tmp/non_existent_dir"])
    resolver._pdf_cache.clear()
    pdf_path = resolver.find_pdf_file("completely_fictional_999999.pdf")
    assert pdf_path is None
