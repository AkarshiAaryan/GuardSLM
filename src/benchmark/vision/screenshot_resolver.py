"""Screenshot Resolver: Maps canonical case references to lossless page PNGs."""

import os
import shutil
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from ..data.schema import ModelInputCase, ScreenshotMapping


class ScreenshotResolver:
    """Locates source PDFs and extracts lossless PNG page images for Strategy B."""

    DEFAULT_SEARCH_PATHS = [
        "data/source_pdfs",
        "/home/regenesis/Desktop/LLM_Project/dataset",
        "/home/regenesis/Desktop/LLM_Project",
        "data",
        "."
    ]

    def __init__(self, search_paths: Optional[List[str]] = None, cache_dir: str = "data/extracted_screenshots"):
        self.search_paths = search_paths or []
        env_path = os.environ.get("GUARDSLM_PDF_DIR")
        if env_path:
            self.search_paths.append(env_path)
        self.search_paths.extend(self.DEFAULT_SEARCH_PATHS)

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._pdf_cache: Dict[str, Optional[str]] = {}

    def find_pdf_file(self, filename: str) -> Optional[str]:
        """Locates a PDF file across search paths, accounting for known naming aliases."""
        if filename in self._pdf_cache:
            return self._pdf_cache[filename]

        candidates = [filename]
        # Common aliases
        if filename.endswith("(1).pdf"):
            candidates.append(filename.replace("(1).pdf", ".pdf"))
        elif filename.endswith(".pdf"):
            candidates.append(filename.replace(".pdf", "(1).pdf"))

        for search_dir in self.search_paths:
            if not os.path.exists(search_dir):
                continue
            for cand in candidates:
                cand_path = os.path.join(search_dir, cand)
                if os.path.isfile(cand_path):
                    self._pdf_cache[filename] = os.path.abspath(cand_path)
                    return self._pdf_cache[filename]

        self._pdf_cache[filename] = None
        return None

    def extract_page_png(self, pdf_path: str, page_number_1_indexed: int, output_png_path: str) -> bool:
        """Extracts a single page as lossless PNG using pdftoppm or python fallbacks."""
        if os.path.isfile(output_png_path) and os.path.getsize(output_png_path) > 0:
            return True

        # Try pdftoppm (fast, crisp, built-in poppler utility)
        pdftoppm_bin = shutil.which("pdftoppm")
        if pdftoppm_bin:
            temp_prefix = output_png_path.replace(".png", "_tmp")
            cmd = [
                pdftoppm_bin,
                "-png",
                "-r", "150",
                "-f", str(page_number_1_indexed),
                "-l", str(page_number_1_indexed),
                pdf_path,
                temp_prefix
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # pdftoppm generates files like <prefix>-1.png or <prefix>-01.png
                dir_name = os.path.dirname(temp_prefix)
                base_name = os.path.basename(temp_prefix)
                for f in os.listdir(dir_name):
                    if f.startswith(base_name) and f.endswith(".png"):
                        src_f = os.path.join(dir_name, f)
                        os.replace(src_f, output_png_path)
                        return True
            except Exception:
                pass

        # Python fallback if fitz (PyMuPDF) or pdf2image is installed
        try:
            import fitz
            doc = fitz.open(pdf_path)
            if 1 <= page_number_1_indexed <= len(doc):
                page = doc.load_page(page_number_1_indexed - 1)
                pix = page.get_pixmap(dpi=150)
                pix.save(output_png_path)
                return True
        except ImportError:
            pass

        return False

    def resolve_case_screenshot(self, case: ModelInputCase) -> Tuple[Optional[str], Dict[str, Any]]:
        """Resolves the verified screenshot for a case.
        
        Returns:
            (screenshot_path_or_None, metadata_dict)
        """
        scrs = case.page_observation.original_pdf_screenshots
        if not scrs:
            return None, {
                "status": "unavailable",
                "error": "No screenshot mapping in case observation",
                "case_id": case.case_id
            }

        primary_scr: ScreenshotMapping = scrs[0]
        filename = primary_scr.source_pdf_filename
        page_num = primary_scr.page_number_1_indexed
        pdf_id = primary_scr.pdf_id

        pdf_path = self.find_pdf_file(filename)
        if not pdf_path:
            return None, {
                "status": "missing_source_pdf",
                "error": f"Source PDF not found: {filename}",
                "pdf_id": pdf_id,
                "case_id": case.case_id,
                "searched_paths": self.search_paths
            }

        cache_png_name = f"{pdf_id}_page_{page_num:02d}.png"
        output_png_path = os.path.join(self.cache_dir, cache_png_name)

        success = self.extract_page_png(pdf_path, page_num, output_png_path)
        if not success or not os.path.isfile(output_png_path):
            return None, {
                "status": "extraction_failed",
                "error": f"Failed to extract page {page_num} from {pdf_path}",
                "case_id": case.case_id
            }

        return output_png_path, {
            "status": "resolved",
            "screenshot_path": os.path.abspath(output_png_path),
            "pdf_id": pdf_id,
            "source_pdf": pdf_path,
            "page_number": page_num,
            "privacy_redaction_required": primary_scr.privacy_redaction_required_before_external_model_call,
            "attachment_key": primary_scr.screenshot_attachment_key
        }
