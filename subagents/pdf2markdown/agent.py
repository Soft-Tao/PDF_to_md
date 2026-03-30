from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from .pdf_converter import PDFConverter
from .vlm_client import QwenVLMClient


class PDF2MarkdownAgent:
    """Converts PDF documents to Markdown using Qwen VLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dpi: int = 200,
        enable_thinking: bool = False,
        thinking_budget: int = 81920,
    ):
        self.pdf_converter = PDFConverter(dpi=dpi)
        self.vlm_client = QwenVLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )

    def convert_pdf(
        self,
        pdf_path: str | Path,
        output_path: Optional[str | Path] = None,
        page_range: Optional[tuple[int, int]] = None,
        stream: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        page_separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Convert a PDF to Markdown.

        Args:
            pdf_path: Input PDF file.
            output_path: Where to save the .md file. If None, returns string only.
            page_range: (start, end) 1-indexed page range. None = all pages.
            stream: Stream API responses to console.
            progress_callback: Called as callback(current, total) for each page.
            page_separator: String inserted between pages (default: horizontal rule).

        Returns:
            Complete Markdown string.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        total_pages = self.pdf_converter.get_page_count(pdf_path)
        start_page = max(1, page_range[0]) if page_range else 1
        end_page = min(total_pages, page_range[1]) if page_range else total_pages
        pages_to_process = end_page - start_page + 1

        print(f"Converting PDF: {pdf_path.name}")
        print(f"Total pages: {total_pages}, Processing pages {start_page}-{end_page}")

        markdown_parts = []
        with tqdm(total=pages_to_process, desc="Converting pages") as pbar:
            for page_num, image in self.pdf_converter.pdf_to_images(
                pdf_path, page_range=(start_page, end_page)
            ):
                if progress_callback:
                    progress_callback(page_num - start_page + 1, pages_to_process)

                print(f"\n--- Processing page {page_num} ---")
                page_markdown = self.vlm_client.convert_image_to_markdown(image, stream=stream)
                markdown_parts.append(f"<!-- Page {page_num} -->\n\n{page_markdown}")
                pbar.update(1)

        header = f"# {pdf_path.stem}\n\n*Converted from: {pdf_path.name}*\n\n---\n\n"
        full_markdown = header + page_separator.join(markdown_parts)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(full_markdown, encoding="utf-8")
            print(f"\nMarkdown saved to: {output_path}")

        return full_markdown

    def convert_single_page(
        self,
        pdf_path: str | Path,
        page_number: int,
        stream: bool = False,
    ) -> str:
        """Convert a single page to Markdown (no header or page markers)."""
        pdf_path = Path(pdf_path)
        for page_num, image in self.pdf_converter.pdf_to_images(
            pdf_path, page_range=(page_number, page_number)
        ):
            return self.vlm_client.convert_image_to_markdown(image, stream=stream)
        raise ValueError(f"Page {page_number} not found in {pdf_path}")


def create_agent(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> PDF2MarkdownAgent:
    """Factory function for PDF2MarkdownAgent. Kwargs: dpi, enable_thinking, thinking_budget."""
    return PDF2MarkdownAgent(api_key=api_key, base_url=base_url, model=model, **kwargs)
