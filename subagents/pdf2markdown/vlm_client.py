import os
from typing import Optional

from openai import OpenAI
from PIL import Image

from .pdf_converter import PDFConverter


class QwenVLMClient:
    DEFAULT_SYSTEM_PROMPT = """**CRITICAL LANGUAGE REQUIREMENT: You MUST output in the SAME language and script as the input document. If the document is in Simplified Chinese (简体中文), output STRICTLY in Simplified Chinese (简体中文) — NEVER convert to Traditional Chinese (繁体中文). If the document is in English, output in English. NEVER switch languages or scripts mid-conversion.**

You are an expert physicist and mathematician specialized in converting handwritten lecture notes to Markdown format with perfect LaTeX transcription.

Your task: Accurately transcribe ALL text and mathematical content from the page image into well-formatted Markdown.

## Mathematical Equations

### Inline Math (single $)
- $E = mc^2$, $F = ma$, $\hbar = h/2\pi$, $x_i$, $A_{\mu\nu}$, $T^{\mu\nu}$
- Greek letters: $\alpha$, $\beta$, $\gamma$, $\Gamma$, $\psi$, $\Psi$, $\phi$, $\Phi$

### Display Math (double $$, on separate lines)
$$
\mathcal{L} = \bar{\psi}(i\gamma^\mu \partial_\mu - m)\psi - \frac{1}{4}F_{\mu\nu}F^{\mu\nu}
$$

### Common LaTeX
- Vectors: $\vec{r}$, $\mathbf{p}$, $\hat{n}$
- Operators: $\hat{H}$, $\nabla$, $\nabla^2$, $\partial_\mu$
- Bra-ket: $\langle \psi |$, $| \phi \rangle$
- Derivatives: $\frac{\partial}{\partial x}$, $\frac{d}{dt}$, $\dot{x}$, $\ddot{x}$
- Integrals: $\int$, $\oint$, $\int_a^b$
- Sums: $\sum_{i=1}^{N}$, $\prod_{k}$
- Matrices: $\begin{pmatrix}a & b \\ c & d\end{pmatrix}$
- Tensor indices: $g_{\mu\nu}$, $\epsilon^{ijk}$
- Symbols: $\hbar$, $\infty$, $\propto$, $\approx$, $\sim$, $\ll$, $\gg$, $\times$, $\cdot$, $\otimes$, $\dagger$

### Equation Numbering
$$
E = mc^2 \tag{1}
$$

## Document Structure
- Use # for title, ## for sections, ### for subsections
- Preserve paragraph breaks and logical flow

## Figures and Tables
- Tables: convert to Markdown table format, preserve all values and units
- Hand-drawn diagrams: write `[图示]` as a placeholder at the location of each diagram; do NOT attempt to describe or reproduce it

## Quality Checks
- Every opening $ must have a closing $
- Every \left must have a matching \right
- Verify all subscripts, superscripts, and Greek letters match the original exactly

OUTPUT ONLY the Markdown content. No preamble, no commentary."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        enable_thinking: bool = False,
        thinking_budget: int = 81920,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not provided. Set DASHSCOPE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        base_url = base_url or os.getenv("VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model or os.getenv("VLM_MODEL", "qwen-vl-max")
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.pdf_converter = PDFConverter()

    def convert_image_to_markdown(
        self,
        image: Image.Image,
        stream: bool = False,
    ) -> str:
        """Convert a rendered PDF page image to Markdown using Qwen VLM."""
        image_url = self.pdf_converter.get_base64_data_url(image, format="JPEG")

        prompt = """Convert this handwritten physics lecture page to Markdown.

Requirements:
1. Transcribe ALL text exactly as shown, preserving Simplified Chinese (简体中文).
2. Convert ALL handwritten formulas and equations to LaTeX (inline $ or display $$).
3. Pay close attention to subscripts, superscripts, Greek letters, and operators.
4. Preserve equation numbers using \\tag{}.
5. Preserve section headings and paragraph structure.
6. For hand-drawn diagrams, write `[图示]` as a placeholder — do not describe them.

Begin transcription:"""

        messages = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        extra_body = {"enable_thinking": True, "thinking_budget": self.thinking_budget} if self.enable_thinking else None

        if stream:
            return self._stream_response(messages, extra_body)
        return self._get_response(messages, extra_body)

    def _get_response(self, messages: list, extra_body) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=extra_body,
        )
        return completion.choices[0].message.content

    def _stream_response(self, messages: list, extra_body) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            extra_body=extra_body,
        )

        content = ""
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content += delta.content
                    print(delta.content, end="", flush=True)

        print()
        return content
