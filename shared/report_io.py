"""Write final tables and fill the report-section template. Owner: Shi Ting."""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd


def to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    """Plain GitHub markdown table (no tabulate dependency)."""
    d = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    d.columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in d.columns]

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:,.{digits}f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


class Exporter:
    """Collects every table and number the report uses, in one output folder."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.tables_dir = os.path.join(out_dir, "tables")
        self.figures_dir = os.path.join(out_dir, "figures")
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        self.numbers: dict = {}
        self.index: list = []

    def table(self, df: pd.DataFrame, name: str, caption: str = "", digits: int = 4) -> pd.DataFrame:
        df.to_csv(os.path.join(self.tables_dir, f"{name}.csv"))
        with open(os.path.join(self.tables_dir, f"{name}.md"), "w") as f:
            f.write((f"**{caption}**\n\n" if caption else "") + to_markdown(df, digits) + "\n")
        self.index.append({"kind": "table", "name": name, "caption": caption})
        return df

    def figure_stem(self, name: str, caption: str = "") -> str:
        self.index.append({"kind": "figure", "name": name, "caption": caption})
        return os.path.join(self.figures_dir, name)

    def number(self, key: str, value, fmt: str = "{:.4f}"):
        self.numbers[key] = fmt.format(value) if isinstance(value, (float, np.floating)) else str(value)

    def finish(self, template_path: str | None = None) -> dict:
        with open(os.path.join(self.out_dir, "report_numbers.json"), "w") as f:
            json.dump(self.numbers, f, indent=1, sort_keys=True)
        pd.DataFrame(self.index).to_csv(os.path.join(self.out_dir, "index.csv"), index=False)
        out = {"numbers": os.path.join(self.out_dir, "report_numbers.json")}
        if template_path and os.path.exists(template_path):
            out["sections"], out["missing"] = fill_template(template_path, self.numbers,
                                                            os.path.join(self.out_dir, "report_sections_filled.md"))
        return out


def fill_template(template_path: str, numbers: dict, out_path: str) -> tuple[str, list]:
    """Replace {{key}} placeholders; unknown keys stay visible as [[MISSING key]]."""
    with open(template_path) as f:
        text = f.read()
    missing = sorted({k for k in re.findall(r"\{\{\s*([\w@.\-+%]+)\s*\}\}", text) if k not in numbers})
    text = re.sub(r"\{\{\s*([\w@.\-+%]+)\s*\}\}", lambda m: numbers.get(m.group(1), f"[[MISSING {m.group(1)}]]"), text)
    with open(out_path, "w") as f:
        f.write(text)
    return out_path, missing
