import base64
import io
from pathlib import Path
from typing import Generator, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image


class PDFConverter:
    """Renders PDF pages to PIL Images for VLM processing."""

    def __init__(self, dpi: int = 200):
        self.dpi = dpi
        self.zoom = dpi / 72

    def get_page_count(self, pdf_path: str | Path) -> int:
        with fitz.open(pdf_path) as doc:
            return len(doc)

    def pdf_to_images(
        self,
        pdf_path: str | Path,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> Generator[Tuple[int, Image.Image], None, None]:
        """
        Yield (page_number, PIL_image) for each page in the PDF.

        Args:
            pdf_path: Path to the PDF file.
            page_range: Optional (start, end) tuple (1-indexed, inclusive).
                        Pages outside this range are skipped without rendering.
        """
        pdf_path = Path(pdf_path)
        start = page_range[0] if page_range else 1
        end = page_range[1] if page_range else None

        with fitz.open(pdf_path) as doc:
            total = len(doc)
            if end is None:
                end = total
            for page_num in range(start - 1, min(end, total)):
                page = doc[page_num]
                mat = fitz.Matrix(self.zoom, self.zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                yield page_num + 1, img

    def get_base64_data_url(self, image: Image.Image, format: str = "JPEG") -> str:
        """Encode a PIL Image as a base64 data URL."""
        buffer = io.BytesIO()
        save_kwargs = {"quality": 85} if format.upper() == "JPEG" else {}
        image.save(buffer, format=format, **save_kwargs)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{b64}"
