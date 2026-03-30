#!/usr/bin/env python3
"""
merge_refine.py — MinerU + LLM Fusion Refinement
=================================================

Combines MinerU's accurate image/layout segmentation with LLM-quality
text and formula recognition from convert_pdf.py.

Pipeline:
  1. Load MinerU _content_list.json (structure) + reference markdown (content)
  2. Step 1: Re-recognize each equation block via VLM (using MinerU's cropped images)
  3. Step 2: Correct text blocks per page using LLM + reference content
  4. Step 3: Reconstruct markdown from corrected blocks
  5. Step 4: Apply pass1 (regex) + pass2 (LLM) from refine_markdown.py

Usage:
    python merge_refine.py --mineru-dir markdown/3.1-磁场的矢势/auto \\
                           --reference markdown/3.1-磁场的矢势/3.1-磁场的矢势.md
    python merge_refine.py --mineru-dir markdown/3.1-磁场的矢势/auto \\
                           --reference markdown/3.1-磁场的矢势/3.1-磁场的矢势.md \\
                           --output markdown/3.1-磁场的矢势/3.1-磁场的矢势_final.md
    python merge_refine.py --mineru-dir markdown/3.1-磁场的矢势/auto \\
                           --reference markdown/3.1-磁场的矢势/3.1-磁场的矢势.md \\
                           --skip-equations    # skip VLM equation re-recognition
"""

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

# ---------------------------------------------------------------------------
# Reuse pass1 / pass2 from refine_markdown.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from refine_markdown import pass1, pass2, get_llm_client

# ---------------------------------------------------------------------------
# VLM client for equation re-recognition
# ---------------------------------------------------------------------------

def _get_vlm_client():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY not set")
    base_url = os.getenv("VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("VLM_MODEL", "qwen-vl-max")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def _image_to_data_url(image_path: Path) -> str:
    """Encode an image file as base64 data URL."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    return f"data:{mime};base64,{data}"


def recognize_equation_image(client: OpenAI, model: str, image_path: Path) -> str:
    """Send a cropped equation image to VLM and return LaTeX string (with $$ wrapping)."""
    data_url = _image_to_data_url(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {
                        "type": "text",
                        "text": (
                            "请将图中的数学公式识别为标准LaTeX代码。"
                            "直接输出LaTeX内容，用$$...$$包裹，不要任何解释或前缀。"
                            "如果是行内公式也用$$包裹。"
                        ),
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Step 1: Re-recognize equation blocks
# ---------------------------------------------------------------------------

def step1_rerecognize_equations(
    content_list: list[dict],
    images_dir: Path,
    vlm_client: OpenAI,
    vlm_model: str,
) -> list[dict]:
    """Replace equation `text` fields using VLM recognition of cropped images."""
    eq_blocks = [b for b in content_list if b.get("type") == "equation" and b.get("img_path")]
    total = len(eq_blocks)
    print(f"Step 1: Re-recognizing {total} equation images via VLM...")

    for i, block in enumerate(eq_blocks, 1):
        img_path = images_dir / block["img_path"]
        if not img_path.exists():
            print(f"  [{i}/{total}] SKIP (image not found): {img_path}")
            continue
        print(f"  [{i}/{total}] {block['img_path']}...", end=" ", flush=True)
        try:
            latex = recognize_equation_image(vlm_client, vlm_model, img_path)
            block["text"] = latex
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")

    return content_list

# ---------------------------------------------------------------------------
# Step 2: Correct text blocks using reference markdown
# ---------------------------------------------------------------------------

TEXT_CORRECTION_SYSTEM = """\
你是一位物理学文稿编辑。你会收到两份内容：
1. 参考内容（由高质量VLM识别）—— 文字和公式较准确
2. MinerU识别的文本块列表（可能有OCR错误）—— 分块准确但文字错误多

你的任务：以参考内容为准，逐块纠正MinerU文本块中的文字OCR错误。

规则：
- 保持原有分块结构，按【块N】格式逐块输出
- 只纠正文字错误（错别字、乱码、明显不通的词语）
- 文本块内已有的行内公式（$...$）保留原样，不要修改公式内容
- 不要从参考内容中添加新的公式块（$$...$$）到文本块里——公式由单独的公式块处理
- 若某块在参考中无对应内容，原样保留
- 不要合并、拆分块，不要添加或删除块
- 只输出纠错后的块内容，不要任何解释
"""


def _parse_corrected_blocks(response: str, n_blocks: int) -> list[str]:
    """Parse LLM response of format 【块N】...content... back to list."""
    results = {}
    # Match 【块N】 followed by content until next 【块 or end
    pattern = re.compile(r'【块(\d+)】([\s\S]*?)(?=【块\d+】|$)')
    for m in pattern.finditer(response):
        idx = int(m.group(1))
        content = m.group(2).strip()
        results[idx] = content
    # Fill in originals for any missing indices (pass-through)
    return results


def correct_text_blocks_for_page(
    llm_client: OpenAI,
    llm_model: str,
    text_blocks: list[dict],
    reference_page: str,
) -> list[dict]:
    """Correct text blocks for one page using the reference content."""
    if not text_blocks:
        return text_blocks

    blocks_prompt = "\n".join(
        f"【块{i+1}】{b['text']}" for i, b in enumerate(text_blocks)
    )
    user_msg = f"参考内容：\n{reference_page}\n\nMinerU文本块：\n{blocks_prompt}"

    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": TEXT_CORRECTION_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content

    corrected_map = _parse_corrected_blocks(raw, len(text_blocks))
    for i, block in enumerate(text_blocks):
        if (i + 1) in corrected_map:
            block["text"] = corrected_map[i + 1]
    return text_blocks


def step2_correct_text_blocks(
    content_list: list[dict],
    reference_pages: dict[int, str],
    llm_client: OpenAI,
    llm_model: str,
) -> list[dict]:
    """Correct all text blocks page by page."""
    # Group text block indices by page
    page_to_text_indices: dict[int, list[int]] = {}
    for idx, block in enumerate(content_list):
        if block.get("type") == "text":
            page = block.get("page_idx", 0)
            page_to_text_indices.setdefault(page, []).append(idx)

    total_pages = len(page_to_text_indices)
    print(f"Step 2: Correcting text blocks across {total_pages} pages via LLM...")

    for page_idx, block_indices in sorted(page_to_text_indices.items()):
        # MinerU page_idx is 0-based; reference_pages keys are 1-based (Page N)
        ref_page = reference_pages.get(page_idx + 1, "")
        if not ref_page:
            print(f"  Page {page_idx}: no reference, skipping")
            continue

        text_blocks = [content_list[i] for i in block_indices]
        n_chars = sum(len(b["text"]) for b in text_blocks)
        print(f"  Page {page_idx}: {len(text_blocks)} text blocks ({n_chars} chars)...", end=" ", flush=True)

        try:
            corrected = correct_text_blocks_for_page(llm_client, llm_model, text_blocks, ref_page)
            for i, block in zip(block_indices, corrected):
                content_list[i] = block
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")

    return content_list

# ---------------------------------------------------------------------------
# Step 3: Reconstruct markdown from corrected content_list
# ---------------------------------------------------------------------------

PAGE_SEP = "\n\n---\n\n"


def _block_to_markdown(block: dict) -> str:
    btype = block.get("type", "text")
    text = block.get("text", "").strip()

    if btype == "image":
        img_path = block.get("img_path", "")
        caption = block.get("img_caption", [])
        caption_text = " ".join(c.get("text", "") for c in caption) if caption else ""
        md = f"![{caption_text}]({img_path})"
        if caption_text:
            md += f"\n\n> {caption_text}"
        return md

    if btype == "equation":
        # text should already be $$...$$ from VLM re-recognition,
        # or the original MinerU latex wrapped in $$
        if text.startswith("$$"):
            return text
        # Fallback: wrap it
        return f"$$\n{text}\n$$"

    if btype == "text":
        level = block.get("text_level")
        if level == 1:
            return f"# {text}"
        if level == 2:
            return f"## {text}"
        if level == 3:
            return f"### {text}"
        return text

    # Unknown type: pass through text
    return text


def step3_reconstruct_markdown(content_list: list[dict], doc_name: str) -> str:
    """Reconstruct the full markdown from the corrected content_list."""
    print("Step 3: Reconstructing markdown...")

    # Group blocks by page
    pages: dict[int, list[dict]] = {}
    for block in content_list:
        page = block.get("page_idx", 0)
        pages.setdefault(page, []).append(block)

    header = f"# {doc_name}\n\n"
    page_parts = []
    for page_idx in sorted(pages.keys()):
        blocks = pages[page_idx]
        block_mds = []
        for block in blocks:
            md = _block_to_markdown(block)
            if md:
                block_mds.append(md)
        page_md = f"<!-- Page {page_idx + 1} -->\n\n" + "\n\n".join(block_mds)
        page_parts.append(page_md)

    return header + PAGE_SEP.join(page_parts)

# ---------------------------------------------------------------------------
# Parse reference markdown into pages
# ---------------------------------------------------------------------------

def parse_reference_pages(reference_md: str) -> dict[int, str]:
    """
    Split reference markdown (from convert_pdf.py) into pages.
    Keys are 1-based page numbers.
    """
    pages = {}
    # Split on <!-- Page N --> markers
    parts = re.split(r'<!--\s*Page\s+(\d+)\s*-->', reference_md)
    # parts: [preamble, page_num, content, page_num, content, ...]
    i = 1
    while i < len(parts) - 1:
        page_num = int(parts[i])
        content = parts[i + 1].strip()
        pages[page_num] = content
        i += 2
    return pages

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge MinerU layout with LLM text quality into refined markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mineru-dir",
        required=True,
        help="MinerU auto/ output directory (contains _content_list.json and images/)",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference markdown from convert_pdf.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <mineru-dir>/../<name>_final.md)",
    )
    parser.add_argument(
        "--skip-equations",
        action="store_true",
        help="Skip VLM re-recognition of equation images (use MinerU's original LaTeX)",
    )
    parser.add_argument(
        "--skip-text-correction",
        action="store_true",
        help="Skip LLM text block correction (keep MinerU's original text)",
    )
    parser.add_argument(
        "--skip-refine",
        action="store_true",
        help="Skip pass1/pass2 post-processing from refine_markdown.py",
    )
    args = parser.parse_args()

    mineru_dir = Path(args.mineru_dir)
    if not mineru_dir.exists():
        print(f"Error: MinerU directory not found: {mineru_dir}", file=sys.stderr)
        sys.exit(1)

    # Find content_list.json
    content_list_files = list(mineru_dir.glob("*_content_list.json"))
    if not content_list_files:
        print(f"Error: No *_content_list.json found in {mineru_dir}", file=sys.stderr)
        sys.exit(1)
    content_list_path = content_list_files[0]
    doc_name = content_list_path.stem.replace("_content_list", "")

    images_dir = mineru_dir  # img_path in content_list already includes "images/" prefix

    reference_path = Path(args.reference)
    if not reference_path.exists():
        print(f"Error: Reference markdown not found: {reference_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else mineru_dir / f"{doc_name}_final.md"

    print(f"MinerU dir:    {mineru_dir}")
    print(f"Reference:     {reference_path}")
    print(f"Output:        {output_path}")
    print(f"Document name: {doc_name}")
    print()

    # Load data
    with open(content_list_path, encoding="utf-8") as f:
        content_list = json.load(f)
    reference_md = reference_path.read_text(encoding="utf-8")
    reference_pages = parse_reference_pages(reference_md)
    print(f"Loaded {len(content_list)} blocks, {len(reference_pages)} reference pages\n")

    # Step 1: Equation re-recognition
    if not args.skip_equations:
        vlm_client, vlm_model = _get_vlm_client()
        print(f"  VLM model: {vlm_model}")
        content_list = step1_rerecognize_equations(content_list, images_dir, vlm_client, vlm_model)
        print()
    else:
        print("Step 1: Skipped (--skip-equations)\n")

    # Step 2: Text block correction
    if not args.skip_text_correction:
        llm_client, llm_model = get_llm_client()
        print(f"  LLM model: {llm_model}")
        content_list = step2_correct_text_blocks(content_list, reference_pages, llm_client, llm_model)
        print()
    else:
        print("Step 2: Skipped (--skip-text-correction)\n")

    # Step 3: Reconstruct markdown
    markdown = step3_reconstruct_markdown(content_list, doc_name)
    print()

    # Step 4: Pass1 + Pass2 from refine_markdown.py
    if not args.skip_refine:
        print("Step 4: Applying refine_markdown pass1 (regex)...")
        markdown = pass1(markdown)
        print("Step 4: pass1 done.")

        print("Step 4: Applying refine_markdown pass2 (LLM)...")
        llm_client, llm_model = get_llm_client()
        print(f"  LLM model: {llm_model}")
        markdown = pass2(markdown, llm_client, llm_model)
        print("Step 4: pass2 done.\n")
    else:
        print("Step 4: Skipped (--skip-refine)\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Done! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
