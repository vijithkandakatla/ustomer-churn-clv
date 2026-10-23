"""Write generated results into README.md between marker comments."""
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def update_readme(results_md: str, readme: Path = Path(__file__).resolve().parents[1] / "README.md") -> None:
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError("README.md is missing the RESULTS markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{results_md.strip()}\n{END}{tail}", encoding="utf-8")


def df_to_markdown(df, floatfmt: str = "{:.3f}") -> str:
    """Small dependency-free markdown table (avoids needing `tabulate`)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
