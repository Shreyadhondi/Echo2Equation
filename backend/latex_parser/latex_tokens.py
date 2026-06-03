# backend/latex_parser/latex_tokens.py

"""
Shared LaTeX token definitions for Echo2Equation.

Why this file exists:
---------------------
MathT5 already has a tokenizer, but spoken-to-LaTeX generation benefits from
teaching the tokenizer common LaTeX commands as meaningful units.

Important:
----------
The same token list must be used during:
1. dataset tokenization
2. model training
3. final inference using the saved tokenizer

If tokenization and training use different vocabularies, token IDs may become
inconsistent. That is why the token list is kept in one shared file.
"""

LATEX_SPECIAL_TOKENS = [
    "\\frac", "\\sqrt", "\\nthroot",
    "\\lim", "\\sum", "\\prod",
    "\\int", "\\iint", "\\iiint", "\\oint",
    "\\infty",

    "\\sin", "\\cos", "\\tan", "\\cot", "\\sec", "\\csc",
    "\\arcsin", "\\arccos", "\\arctan",
    "\\sinh", "\\cosh", "\\tanh",
    "\\log", "\\ln",

    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta",
    "\\eta", "\\theta", "\\iota", "\\kappa", "\\lambda", "\\mu",
    "\\nu", "\\xi", "\\pi", "\\rho", "\\sigma", "\\tau",
    "\\upsilon", "\\phi", "\\chi", "\\psi", "\\omega",

    "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Xi", "\\Pi",
    "\\Sigma", "\\Upsilon", "\\Phi", "\\Psi", "\\Omega",

    "\\pm", "\\mp", "\\cdot", "\\times", "\\div",
    "\\ast", "\\star", "\\circ", "\\bullet",
    "\\oplus", "\\ominus", "\\otimes", "\\oslash", "\\odot",

    "\\leq", "\\geq", "\\neq", "\\approx", "\\equiv", "\\sim",
    "\\propto", "\\cong",

    "\\subset", "\\supset", "\\subseteq", "\\supseteq",
    "\\in", "\\ni", "\\notin",
    "\\cup", "\\cap", "\\emptyset", "\\varnothing",

    "\\rightarrow", "\\leftarrow", "\\leftrightarrow",
    "\\Rightarrow", "\\Leftarrow", "\\Leftrightarrow",
    "\\longrightarrow", "\\longleftarrow",
    "\\Longrightarrow", "\\Longleftarrow",
    "\\mapsto", "\\to",

    "\\left", "\\right",
    "\\langle", "\\rangle",
    "\\lfloor", "\\rfloor",
    "\\lceil", "\\rceil",

    "\\partial", "\\nabla", "\\mathrm", "\\operatorname",
    "\\mathbb", "\\mathcal", "\\mathbf",

    "\\forall", "\\exists", "\\nexists",
    "\\neg", "\\land", "\\lor", "\\implies", "\\iff",

    "\\hat", "\\bar", "\\overline", "\\underline",
    "\\tilde", "\\vec",

    "\\begin", "\\end",
    "\\text",

    "^", "_", "{", "}",
]


def get_latex_special_tokens() -> list[str]:
    """
    Return a stable, duplicate-free list of LaTeX special tokens.

    sorted(set(...)) is used so the token order remains deterministic.
    Deterministic ordering is important for reproducibility.
    """
    return sorted(set(LATEX_SPECIAL_TOKENS))


def add_latex_tokens(tokenizer) -> int:
    """
    Add LaTeX special tokens to a HuggingFace tokenizer.

    Returns:
        Number of newly added tokens.
    """
    tokens = get_latex_special_tokens()
    return tokenizer.add_tokens(tokens)
