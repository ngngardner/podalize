"""Document Generator module for creating LaTeX documents with pylatex."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pylatex import (
    Command,
    Document,
    Figure,
    NewLine,
    NewPage,
    Section,
    Subsection,
    Tabular,
)
from pylatex.utils import NoEscape

if TYPE_CHECKING:
    import pandas as pd


class DocumentGenerator:
    """Document Generator class."""

    def __init__(
        self: DocumentGenerator,
        title: str,
        author: str,
        path2logs: Path,
    ) -> None:
        """Initialize the Document Generator class."""
        self.path2logs = path2logs
        self.fig_count = 0
        self.doc = Document()
        self.doc.preamble.append(Command("title", title))
        self.doc.preamble.append(Command("author", author))
        self.doc.preamble.append(Command("date", NoEscape(r"\today")))
        self.doc.append(NoEscape(r"\maketitle"))

    def add_section(
        self: DocumentGenerator,
        section_title: str,
        section_content: str = "",
    ) -> None:
        """Add a section to the document."""
        with self.doc.create(Section(section_title)):
            self.doc.append(section_content)

    def add_sub_section(
        self: DocumentGenerator,
        ss_title: str,
        ss_content: str,
    ) -> None:
        """Add a subsection to the document."""
        with self.doc.create(Subsection(ss_title)):
            self.doc.append(ss_content)

    def add_image(
        self: DocumentGenerator,
        filename: str,
        caption: str = "",
        width: str = "400px",
    ) -> None:
        """Add an image to the document."""
        filepath = Path(filename).absolute()
        with self.doc.create(Figure(position="h!")) as pic:
            self.doc.append(Command("centering"))
            pic.add_image(filepath, width=width)
            pic.add_caption(caption)
        self.fig_count += 1

    def add_new_page(self: DocumentGenerator) -> None:
        """Add a new page to the document."""
        self.doc.append(NewPage())

    def add_pandas_table(self: DocumentGenerator, df: pd.DataFrame) -> None:
        """Add a pandas DataFrame to the document."""
        nr, nc = df.shape
        with self.doc.create(
            Tabular("c" * (nc + 1), pos="centering", row_height=2),
        ) as table:
            table.add_hline()
            table.add_row(["", *list(df.columns)])
            table.add_hline()
            for row in df.index:
                table.add_row([row, *list(df.loc[row, :])])
            table.add_hline()

    def add_new_lines(self: DocumentGenerator, n: int = 1) -> None:
        """Add n new lines to the document."""
        for _ in range(n):
            self.doc.append(NewLine())
