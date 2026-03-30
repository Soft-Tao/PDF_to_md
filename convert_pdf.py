#!/usr/bin/env python3
"""
PDF to Markdown Converter - Convenience Script

Convert scientific PDF documents to Markdown using Qwen VLM.
This is Phase 0 of the rise-agent workflow.

Usage:
    python convert_pdf.py <pdf_file> [options]

Examples:
    python convert_pdf.py document\3.10\1.5-电磁规律中的守恒律-讲义.pdf
    python convert_pdf.py input/paper.pdf
    python convert_pdf.py input/paper.pdf -o markdown/paper.md
    python convert_pdf.py input/paper.pdf --pages 1-5
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from subagents.pdf2markdown import create_agent


def parse_page_range(page_range_str: str) -> tuple[int, int]:
    """Parse page range string like '1-5' into tuple (1, 5)."""
    if "-" in page_range_str:
        start, end = page_range_str.split("-")
        return int(start.strip()), int(end.strip())
    else:
        page = int(page_range_str.strip())
        return page, page


def main():
    parser = argparse.ArgumentParser(
        description="Convert scientific PDF documents to Markdown using Qwen VLM (Phase 0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s input/paper.pdf                     # Convert entire PDF
    %(prog)s input/paper.pdf -o markdown/out.md  # Save to specific file
    %(prog)s input/paper.pdf --pages 1-5         # Convert pages 1 to 5
    %(prog)s input/paper.pdf --thinking          # Enable reasoning mode
        """
    )

    parser.add_argument(
        "pdf_file",
        type=str,
        help="Path to the PDF file to convert"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output Markdown file path (default: markdown/<pdf_name>.md)"
    )

    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page range to convert, e.g., '1-5' or '3' (default: all pages)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Qwen VLM model to use (default: VLM_MODEL env var, or qwen-vl-max)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF rendering (default: 200)"
    )

    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking/reasoning mode (automatically uses qwen3-vl-plus model)"
    )

    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=81920,
        help="Max tokens for thinking process (default: 81920)"
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the output in real-time"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="DashScope API key (default: uses DASHSCOPE_API_KEY env var)"
    )

    args = parser.parse_args()

    # Validate PDF file
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    if not pdf_path.suffix.lower() == ".pdf":
        print(f"Warning: File may not be a PDF: {pdf_path}", file=sys.stderr)

    # Determine output path (default to markdown/ directory)
    output_path = args.output
    if output_path is None:
        output_path = Path("markdown") / pdf_path.stem / f"{pdf_path.stem}.md"

    # Parse page range
    page_range = None
    if args.pages:
        try:
            page_range = parse_page_range(args.pages)
        except ValueError:
            print(f"Error: Invalid page range format: {args.pages}", file=sys.stderr)
            print("Use format like '1-5' or '3'", file=sys.stderr)
            sys.exit(1)

    # Auto-switch to qwen3-vl-plus when thinking is enabled
    model = args.model or os.getenv("VLM_MODEL", "qwen-vl-max")
    if args.thinking and model != "qwen3-vl-plus":
        print(f"Note: Thinking mode requires qwen3-vl-plus, switching from {model}")
        model = "qwen3-vl-plus"

    # Check API key
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("Error: DASHSCOPE_API_KEY not set.", file=sys.stderr)
        print("Please set it in .env file or pass via --api-key", file=sys.stderr)
        sys.exit(1)

    # Create agent and convert
    try:
        agent = create_agent(
            api_key=api_key,
            base_url=os.getenv("VLM_BASE_URL"),
            model=model,
            dpi=args.dpi,
            enable_thinking=args.thinking,
            thinking_budget=args.thinking_budget,
        )

        markdown = agent.convert_pdf(
            pdf_path=pdf_path,
            output_path=output_path,
            page_range=page_range,
            stream=args.stream,
        )

        print(f"\nPhase 0 complete!")
        print(f"  Output: {output_path}")
        print(f"\nNext: Proceed to Phase 1 (Comprehension) - read the markdown and identify figures.")

    except Exception as e:
        print(f"\nError during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
