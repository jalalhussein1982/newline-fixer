from newline_fixer.data.wikipedia import clean_wikipedia_text

ARTICLE = (
    "Anarchism is a political philosophy. It questions the legitimacy of hierarchy and "
    "proposes voluntary association in its place.\n\n"
    "It is skeptical of authority. Its advocates have differed on tactics, on economics "
    "and on how far the critique of hierarchy should extend.\n\n"
    "Etymology\n\n"
    "The word comes from Greek. It entered English in the sixteenth century and was used "
    "as a term of abuse long before anyone claimed it.\n\n"
    "See also\n\n"
    "Libertarianism\nMutualism\n\n"
    "References\n\n"
    "Some citation.\n"
)


def test_clean_wikipedia_drops_trailing_sections_and_keeps_body() -> None:
    out = clean_wikipedia_text(ARTICLE)
    assert out is not None
    assert out.startswith("Anarchism is a political philosophy. It questions")
    assert "\n\nIt is skeptical of authority." in out
    assert "Etymology\n\nThe word comes from Greek." in out
    assert "See also" not in out and "References" not in out


def test_clean_wikipedia_truncates_on_paragraph_boundary() -> None:
    long = "\n\n".join(f"Paragraph {i} " + "text " * 100 for i in range(30))
    out = clean_wikipedia_text(long, max_chars=2000)
    assert out is not None and len(out) <= 2000
    assert out.endswith(out.split("\n\n")[-1])
    assert all(p.startswith("Paragraph") for p in out.split("\n\n"))


def test_clean_wikipedia_rejects_short() -> None:
    assert clean_wikipedia_text("Too short.") is None
