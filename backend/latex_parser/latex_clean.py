import re

def clean_latex(s: str, keep_dollars: bool = False) -> str:
    """
    Normalize common artifacts from seq2seq LaTeX generation.

    If keep_dollars=True, the cleaned string is returned wrapped in $...$.
    (We always strip any existing outer dollars first to avoid $$...$$.)
    """
    if not s:
        return s

    s = s.strip()
    had_dollars = s.startswith("$") and s.endswith("$")
    if had_dollars:
        s = s[1:-1].strip()

    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)

    # Remove spaces immediately after backslash: "\ sum" -> "\sum"
    s = re.sub(r"\\\s+", r"\\", s)

    # Tighten braces: "{ x"->"{x}", "x }"->"x}"
    s = re.sub(r"\{\s+", "{", s)
    s = re.sub(r"\s+\}", "}", s)

    # Remove spaces around ^ and _
    s = re.sub(r"\s*\^\s*", "^", s)
    s = re.sub(r"\s*_\s*", "_", s)

    # Brace single-char exponents/subscripts if not already braced
    s = re.sub(r"\^([A-Za-z0-9])", r"^{\1}", s)
    s = re.sub(r"_([A-Za-z0-9])", r"_{\1}", s)

    # Common spaced commands
    for wrong, right in (
        (r"\\ frac",  r"\\frac"),
        (r"\\ times", r"\\times"),
        (r"\\ div",   r"\\div"),
        (r"\\ log",   r"\\log"),
        (r"\\ ln",    r"\\ln"),
        (r"\\ sin",   r"\\sin"),
        (r"\\ cos",   r"\\cos"),
        (r"\\ tan",   r"\\tan"),
        (r"\\ sum",   r"\\sum"),
        (r"\\ int",   r"\\int"),
        (r"\\ sqrt",  r"\\sqrt"),
        (r"\\ pi",    r"\\pi"),
    ):
        s = re.sub(wrong, right, s)

    # Best-effort brace balance (add missing closers only)
    opens, closes = s.count("{"), s.count("}")
    if opens > closes:
        s += "}" * (opens - closes)

    s = s.strip()

    if keep_dollars:
        # ensure we don't produce $$...$$
        s = s.strip().strip("$")
        s = f"${s}$"

    return s
