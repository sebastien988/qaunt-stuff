"""The website builder: which chart belongs to which script, and what ends up on the page."""

from types import SimpleNamespace

import pytest

import publish


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """An empty project folder that publish.py treats as its own."""
    monkeypatch.setattr(publish, "FOLDER", tmp_path)
    monkeypatch.setattr(publish, "SITE", tmp_path / "docs")
    monkeypatch.setattr(publish, "CHARTS", tmp_path / "docs" / "charts")
    return tmp_path


def write_script(folder, name: str, docstring: str, body: str = "") -> None:
    (folder / name).write_text(f'"""{docstring}"""\n{body}\n')


def test_docstring_summary(folder):
    write_script(folder, "a.py", "Rank things by strength.\n\nThe score is an\naverage of returns.\n\nUsage:\n    python a.py\n")
    assert publish.docstring_summary(folder / "a.py") == ("Rank things by strength", "The score is an average of returns.")

    write_script(folder, "b.py", "Only a title.\n\nUsage:\n    python b.py\n")
    assert publish.docstring_summary(folder / "b.py") == ("Only a title", "")

    (folder / "no_docstring.py").write_text("x = 1\n")
    assert publish.docstring_summary(folder / "no_docstring.py") == ("No docstring", "")
    (folder / "broken.py").write_text("def (:\n")
    assert publish.docstring_summary(folder / "broken.py") == ("Broken", "")


def test_each_chart_is_listed_once_under_the_script_that_makes_it(folder):
    write_script(folder, "bond_scan.py", "Scan bonds.", 'CHART = "bond_scan.png"')
    write_script(folder, "market_scan.py", "Scan markets.", 'CHARTS = ["market_scan.png", "market_top10.png"]')
    # A script that only mentions another script's chart must not claim it.
    write_script(folder, "compare_chart.py", "Compare.", 'OUT = "comparison.png"  # see also bond_scan.png')
    write_script(folder, "data_tool.py", "No chart at all.")
    write_script(folder, "publish.py", "Not an algorithm.", '"bond_scan.png"')
    (folder / "market_scan.md").write_text("report")
    for name in ("bond_scan.png", "market_scan.png", "market_top10.png", "comparison.png", "orphan_chart.png"):
        (folder / name).write_bytes(b"png")

    found = publish.find_algorithms()
    by_script = {a["script"].name if a["script"] else None: a for a in found}
    charts = lambda script: sorted(c.name for c in by_script[script]["charts"])
    assert charts("bond_scan.py") == ["bond_scan.png"]
    assert charts("market_scan.py") == ["market_scan.png", "market_top10.png"]
    assert charts("compare_chart.py") == ["comparison.png"]
    assert charts("data_tool.py") == []
    assert charts(None) == ["orphan_chart.png"]  # a chart no script mentions is still shown
    assert "publish.py" not in by_script
    assert by_script["market_scan.py"]["report"].name == "market_scan.md"
    assert by_script["bond_scan.py"]["report"] is None

    listed = [c.name for a in found for c in a["charts"]]
    assert sorted(listed) == sorted(set(listed))  # no chart appears twice


@pytest.mark.parametrize("remote, expected", [
    ("https://github.com/someone/some-repo.git", ("someone", "some-repo")),
    ("https://github.com/someone/some-repo", ("someone", "some-repo")),
    ("git@github.com:someone/some.repo.git", ("someone", "some.repo")),
    ("https://example.com/someone/some-repo.git", None),
    ("", None),
])
def test_repository_is_read_from_the_git_remote(monkeypatch, remote, expected):
    monkeypatch.setattr(publish, "git", lambda *args, **kwargs: SimpleNamespace(stdout=remote + "\n", returncode=0))
    assert publish.repository() == expected


def test_site_and_readme_list_every_chart(folder):
    write_script(folder, "bond_scan.py", "Scan <bonds> & more.\n\nYield and risk.", '"bond_scan.png"')
    write_script(folder, "data_tool.py", "No chart at all.")
    (folder / "bond_scan.png").write_bytes(b"png")
    (folder / "bond_scan.md").write_text("report")
    algorithms = publish.find_algorithms()

    assert publish.build_site(algorithms, ("someone", "some-repo")) == 1
    page = (folder / "docs" / "index.html").read_text()
    assert (folder / "docs" / "charts" / "bond_scan.png").read_bytes() == b"png"
    assert (folder / "docs" / ".nojekyll").exists()
    assert "Scan &lt;bonds&gt; &amp; more" in page and "<bonds>" not in page  # text is escaped
    assert 'src="charts/bond_scan.png"' in page
    assert "https://github.com/someone/some-repo/blob/main/bond_scan.py" in page
    assert "https://github.com/someone/some-repo/blob/main/bond_scan.md" in page
    assert "No chart at all" not in page  # scripts without a chart get no card
    assert publish.DISCLAIMER.split(";")[0] in page

    # Charts deleted from the folder disappear from the site on the next build.
    (folder / "bond_scan.png").unlink()
    assert publish.build_site(publish.find_algorithms(), None) == 0
    assert not (folder / "docs" / "charts" / "bond_scan.png").exists()

    publish.build_readme(algorithms, ("someone", "some-repo"))
    readme = (folder / "README.md").read_text()
    assert "https://someone.github.io/some-repo/" in readme
    assert "| [`bond_scan.py`](bond_scan.py) | Scan <bonds> & more | [bond_scan.png](bond_scan.png) | [bond_scan.md](bond_scan.md) |" in readme
    assert "| [`data_tool.py`](data_tool.py) | No chart at all | - | - |" in readme
    assert "python -m pytest" in readme

    publish.build_readme(algorithms, None)
    assert "docs/index.html" in (folder / "README.md").read_text()
