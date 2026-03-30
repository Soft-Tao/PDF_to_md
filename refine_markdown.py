"""
Markdown Refinement Script for Physics Lecture Notes
=====================================================

Post-processes AI-converted Markdown files from handwritten physics PDFs.

Rules applied:
  Pass 1 (regex):
    3. Demote ## / ### headings to **bold** (keep only one # title)
    6. Replace English punctuation with Chinese outside formula environments
       Add/normalize single space around inline $...$ formulas
    2 (partial). Replace \\ell with l in all formula regions

  Pass 2 (LLM):
    1. \vec{}, \hat{}, \overleftrightarrow{} normalization
    2. \mathrm{d} for differential operator
    4. Fix obvious text OCR errors
    5. Fix obvious formula OCR errors

Usage:
    python refine_markdown.py <input.md> [output.md]

    If output.md is omitted, writes to <input_stem>_refined.md in the same directory.
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
你是一位专业的物理学LaTeX编辑。请对输入的Markdown文本**只**做以下修改，其他内容一字不改地原样输出：

1. **矢量符号**：将公式中表示矢量的符号统一改为 \\vec{}。例如 \\mathbf{E} → \\vec{E}，\\boldsymbol{B} → \\vec{B}。\
   注意：只改表示物理矢量的符号（如场强、速度、位移等），不要把矩阵或其他非矢量粗体改掉。

2. **单位矢量 / 帽子符号**：将表示单位矢量或算符帽的符号统一改为 \\hat{}。例如 \\hat{n}、\\hat{H}、\\hat{p} 应已正确；\
   若原来用 \\mathbf{\\hat{}}、\\boldsymbol{\\hat{}} 等，改为 \\hat{}。

3. **张量符号**：将表示张量的符号（通常带双箭头或用 \\overleftrightarrow 标注）统一改为 \\overleftrightarrow{}。

4. **全微分符号**：将公式中用作微分算符的裸字母 d 改为 \\mathrm{d}。具体包括：
   - 积分末尾的微元：dx → \\mathrm{d}x，dV → \\mathrm{d}V，d^3r → \\mathrm{d}^3r，d\\vec{S} → \\mathrm{d}\\vec{S} 等
   - 导数分子：\\frac{d}{dt} → \\frac{\\mathrm{d}}{\\mathrm{d}t}，\\frac{df}{dx} → \\frac{\\mathrm{d}f}{\\mathrm{d}x} 等
   - 注意：变量名中的 d（如 \\delta、d_{ij} 下标中的字母）不要改。

5. **字母右上角的撇号**：如果你看到某个字母（如 r、V、S 等）右上方有一个撇号（'），很可能是表示源点的物理量，请统一使用{}^{\\prime}的LaTeX格式。例如 r' → r^{\\prime}，V' → V^{\\prime}，S' → S^{\\prime}

6. **文字OCR错误**：如果你非常确定某处中文文字识别有误（如明显的错字、乱码、上下文明显不通），直接改正。\
   不确定的不要改。

7. **公式OCR错误**：如果你非常确定某个LaTeX公式有识别错误（如缺少大括号、明显的符号错误），直接改正。\
   不确定的不要改。

8. **积分域下标补全**：在含积分符号的公式块（`$$...$$`）或连等式中，若某个积分已标注了积分域下标（如 `\\int_{V}`、`\\int_{S}`、`\\oint_{S^{\\prime}}` 等），则根据物理意义补全同一连等式中其他积分符号的下标。具体规则：
   - 同一连等式中，同类积分（体积分、面积分、线积分）若部分有下标部分没有，按物理一致性补全。
   - 高斯定理变换：体积分 `\\int_{V} \\nabla \\cdot \\vec{F} \\, \\mathrm{d}V` 对应面积分 `\\oint_{S}`（S 为 V 的边界面）；反之亦然。
   - 斯托克斯定理变换：面积分 `\\int_{S} (\\nabla \\times \\vec{F}) \\cdot \\mathrm{d}\\vec{S}` 对应线积分 `\\oint_{L}`（L 为 S 的边界线）；反之亦然。
   - 源点积分（带 prime 的域，如 `V^{\\prime}`、`S^{\\prime}`）与场点积分（不带 prime）分别对应，不要混用。
   - 若某积分的域在物理上不明确（无法从上下文确定），不要擅自添加下标。

**重要约束**：
- 不要改变文档结构（标题、图片说明等）
- 不要添加或删除任何内容（规则 8 的下标补全除外）
- 不要修改图片说明块（以 > 开头的引用块）中的内容
- 只输出修改后的Markdown，不要加任何解释或前缀
"""


def get_llm_client():
    # REFINE_API_KEY / REFINE_BASE_URL let you use a different provider for
    # text refinement (e.g. Claude via aihubmix) while VLM stays on DashScope.
    api_key = os.getenv("REFINE_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Neither REFINE_API_KEY nor DASHSCOPE_API_KEY is set")
    base_url = (
        os.getenv("REFINE_BASE_URL")
        or os.getenv("VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    model = os.getenv("REFINE_MODEL") or os.getenv("VLM_MODEL", "qwen-plus")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


# ---------------------------------------------------------------------------
# Pass 1: Regex transformations
# ---------------------------------------------------------------------------

def _iter_segments(text: str):
    """
    Yield (segment_text, is_formula) tuples by parsing the markdown.

    Formula regions:
      - Display math:  $$...$$  (can be multi-line)
      - Inline math:   $...$
      - Code fences:   ```...```
      - HTML comments: <!-- ... -->

    Everything else is a text region.
    """
    pattern = re.compile(
        r'(\$\$[\s\S]*?\$\$'        # display math $$...$$
        r'|\$[^\$\n]+?\$'           # inline math $...$
        r'|```[\s\S]*?```'          # code fence
        r'|<!--[\s\S]*?-->'         # html comment
        r'|!\[[^\]]*\]\([^)]*\))',  # markdown image ![alt](url)
        re.DOTALL,
    )
    pos = 0
    for m in pattern.finditer(text):
        start, end = m.start(), m.end()
        if start > pos:
            yield text[pos:start], False   # text region
        yield text[start:end], True        # formula / special region
        pos = end
    if pos < len(text):
        yield text[pos:], False            # trailing text


# English → Chinese punctuation pairs (applied in text regions only)
_PUNCT_MAP = [
    (',', '，'),
    ('(', '（'),
    (')', '）'),
    (':', '：'),
    (';', '；'),
    ('?', '？'),
    ('!', '！'),
]


def _replace_punct_in_text(text: str) -> str:
    """Replace English punctuation with Chinese in a pure-text segment."""
    for en, zh in _PUNCT_MAP:
        text = text.replace(en, zh)
    # Period: only replace when not preceded/followed by digit or ASCII letter
    # (preserves "3.14", "Fig. 1", URLs, etc.)
    text = re.sub(r'(?<![0-9a-zA-Z])\.(?![0-9a-zA-Z])', '。', text)
    return text


def _demote_headings(text: str) -> str:
    """
    Demote ## and deeper headings to bold paragraphs.
    The single # title line is left untouched.
    """
    def replacer(m):
        hashes = m.group(1)
        title = m.group(2).strip()
        if len(hashes) == 1:
            return m.group(0)  # keep # title
        return f'\n**{title}**'

    return re.sub(r'^(#{1,6})\s+(.+)$', replacer, text, flags=re.MULTILINE)


def _normalize_inline_formula_spacing(text: str) -> str:
    """
    Ensure exactly one space on each side of inline $...$ formulas
    when adjacent to non-whitespace text.
    """
    # Add space before $ if preceded by non-space (but not another $)
    text = re.sub(r'(?<=\S)(\$(?!\$))', r' \1', text)
    # Add space after closing $ if followed by non-space (but not another $)
    text = re.sub(r'(?<!\$)(\$)(?=\S)', r'\1 ', text)
    return text


def _replace_ell_in_formula(formula: str) -> str:
    """Replace \\ell with l inside formula regions."""
    return formula.replace(r'\ell', 'l')


def _remove_page_markers(text: str) -> str:
    """Remove page separator markers inserted by the converter.

    Removes:
      - <!-- Page N --> HTML comment markers
      - Standalone --- horizontal rules used as page separators
        (a line containing only --- that is preceded/followed by blank lines)
    """
    # Remove <!-- Page N --> (with optional surrounding whitespace/newlines)
    text = re.sub(r'\n*<!--\s*Page\s+\d+\s*-->\n*', '\n\n', text)
    # Remove standalone --- page separators (preceded and followed by blank lines)
    text = re.sub(r'\n\n---\n\n', '\n\n', text)
    # Clean up excessive blank lines left behind (more than 2 consecutive newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def pass1(text: str) -> str:
    """Apply all regex-based transformations."""
    # Step 0: remove page markers and separators
    text = _remove_page_markers(text)

    # Step 1: demote headings (line-by-line, safe for whole text)
    text = _demote_headings(text)

    # Step 2: segment-aware punctuation + \ell replacement
    segments = list(_iter_segments(text))
    result_parts = []
    for seg, is_formula in segments:
        if is_formula:
            seg = _replace_ell_in_formula(seg)
        else:
            seg = _replace_punct_in_text(seg)
        result_parts.append(seg)
    text = ''.join(result_parts)

    # Step 3: normalize inline formula spacing
    text = _normalize_inline_formula_spacing(text)

    return text


# ---------------------------------------------------------------------------
# Pass 2: LLM transformations
# ---------------------------------------------------------------------------

# Page separator used by the converter
PAGE_SEP = '\n\n---\n\n'

# Max chars per chunk sent to LLM
MAX_CHUNK_CHARS = 4000


def _split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split text at page separators. If a single page exceeds max_chars,
    split further at paragraph breaks.
    """
    pages = text.split(PAGE_SEP)
    chunks = []
    for page in pages:
        if len(page) <= max_chars:
            chunks.append(page)
        else:
            paragraphs = page.split('\n\n')
            current = ''
            for para in paragraphs:
                if current and len(current) + len(para) + 2 > max_chars:
                    chunks.append(current)
                    current = para
                else:
                    current = (current + '\n\n' + para) if current else para
            if current:
                chunks.append(current)
    return chunks


def _llm_refine_chunk(client: OpenAI, model: str, chunk: str) -> str:
    """Send one chunk to the LLM and return the refined text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ],
    )
    return response.choices[0].message.content


def pass2(text: str, client: OpenAI, model: str) -> str:
    """Apply LLM-based transformations chunk by chunk."""
    chunks = _split_into_chunks(text)
    total = len(chunks)
    refined_chunks = []

    for i, chunk in enumerate(chunks, 1):
        print(f"  LLM refining chunk {i}/{total} ({len(chunk)} chars)...", end=' ', flush=True)
        refined = _llm_refine_chunk(client, model, chunk)
        refined_chunks.append(refined)
        print("done")

    return PAGE_SEP.join(refined_chunks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def refine(input_path: Path, output_path: Path) -> None:
    text = input_path.read_text(encoding='utf-8')

    print(f"Pass 1: regex transformations...")
    text = pass1(text)
    print(f"Pass 1 done.")

    print(f"Pass 2: LLM transformations...")
    client, model = get_llm_client()
    print(f"  Using model: {model}")
    text = pass2(text, client, model)
    print(f"Pass 2 done.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding='utf-8')
    print(f"\nRefined markdown saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python refine_markdown.py <input.md> [output.md]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.with_stem(input_path.stem + '_refined')

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    refine(input_path, output_path)


if __name__ == '__main__':
    main()
