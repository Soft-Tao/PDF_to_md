"""
PDF to Markdown Converter Package
==================================

A specialized tool for converting scientific PDF documents (including handwritten content)
to well-structured Markdown format with LaTeX equation preservation.

Author: RISE-AGI, Peking University
Purpose: Enable reproducible physics research by digitizing papers with full equation fidelity

Components:
    - PDF2MarkdownAgent: Main orchestration agent for full conversion pipeline
    - PDFConverter: PDF page rendering to high-resolution images (PyMuPDF)
    - QwenVLMClient: Image-to-Markdown conversion using Qwen Vision Language Model

Key Features:
    ✓ LaTeX equation preservation (inline $ and display $$)
    ✓ Figure metadata extraction for reproducibility
    ✓ Table structure recognition
    ✓ Handwritten content support (high DPI mode)
    ✓ Batch processing and partial page ranges
    ✓ Streaming and non-streaming API responses

Quick Start:
    >>> from subagents.pdf2markdown import create_agent
    >>> agent = create_agent()  # Uses environment variables
    >>> agent.convert_pdf(
    ...     pdf_path="input/paper.pdf",
    ...     output_path="markdown/paper.md"
    ... )

Environment Variables:
    - DASHSCOPE_API_KEY (required): DashScope API authentication
    - VLM_BASE_URL (optional): API endpoint (default: DashScope)
    - VLM_MODEL (optional): Model name (default: qwen-vl-max)

For detailed usage, see README.md in this directory.
"""

from .agent import PDF2MarkdownAgent, create_agent
from .pdf_converter import PDFConverter
from .vlm_client import QwenVLMClient

__all__ = [
    "PDF2MarkdownAgent",
    "create_agent",
    "PDFConverter",
    "QwenVLMClient",
]
