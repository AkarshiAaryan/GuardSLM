"""Test screenshot resolver, PDF alias mapping, and text-only non-resolution."""

import os
from src.benchmark.data.loader import BenchmarkLoader
from src.benchmark.vision.screenshot_resolver import ScreenshotResolver


def test_screenshot_resolution_text_only_case():
    loader = BenchmarkLoader()
    cases = loader.get_model_inputs(limit=1)
    resolver = ScreenshotResolver()

    # In pure text-only benchmark, cases do not embed screenshots
    scr_path, meta = resolver.resolve_case_screenshot(cases[0])
    assert scr_path is None
    assert meta["status"] == "unavailable"


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
