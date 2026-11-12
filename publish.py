"""Build the website from every chart, then upload everything to GitHub.

Run this after adding or re-running an algorithm:

    python publish.py --repo USERNAME/REPOSITORY    # first time: says which GitHub repository to use
    python publish.py                               # every time after that
    python publish.py --no-push                     # build and commit only

What it does
  1. Finds every .png in this folder and the .py script that produces it (the
     script whose code mentions the file name). Nothing needs registering: a
     new algorithm's chart appears on the site the next time this runs.
  2. Writes the website into docs/ (docs/index.html plus copies of the charts),
     which is where GitHub Pages serves a site from.
  3. Rewrites README.md so the repository's front page advertises the website
     and lists every algorithm.
  4. Commits all changes and pushes them to GitHub.

Needs only the Python standard library and git.
"""

import argparse
import ast
import html
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
SITE = FOLDER / "docs"
CHARTS = SITE / "charts"
TITLE = "Quant Research"
TAGLINE = "Trading algorithms, market scans and forecasts, each tested on past data and reported honestly."
DISCLAIMER = ("Educational research, not financial advice. Charts describe the past and statistical forecasts; "
              "most of the forecasts here did not beat simple baselines, and the reports say so.")
NOT_ALGORITHMS = {"publish.py", "main.py", "claculator.py"}
IGNORE = """.venv/
__pycache__/
build/
.vscode/
*_cache.csv
*_cache.json
"""
REQUIREMENTS = "numpy\npandas\nmatplotlib\nyfinance\nrequests\n"


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=FOLDER, text=True, capture_output=True, check=check)


def docstring_summary(script: Path) -> tuple[str, str]:
    """First line of the script's docstring, and the paragraph after it."""
    try:
        doc = ast.get_docstring(ast.parse(script.read_text())) or ""
    except SyntaxError:
        doc = ""
    parts = [" ".join(p.split()) for p in doc.split("\n\n") if p.strip()]
    title = parts[0].rstrip(".") if parts else script.stem.replace("_", " ").capitalize()
    detail = parts[1] if len(parts) > 1 and not parts[1].lower().startswith("usage") else ""
    return title, detail


def find_algorithms() -> list[dict]:
    """Every script, with the charts and report it produces. Newest chart first."""
    scripts = sorted(p for p in FOLDER.glob("*.py") if p.name not in NOT_ALGORITHMS)
    sources = {p: p.read_text() for p in scripts}
    charts = sorted(FOLDER.glob("*.png"))
    found = []
    for script in scripts:
        mine = [c for c in charts if c.name in sources[script]]
        # A chart mentioned by several scripts belongs to the one named most like it.
        mine = [c for c in mine if not any(
            other != script and c.name in text and other.stem.split("_")[0] == c.stem.split("_")[0]
            and script.stem.split("_")[0] != c.stem.split("_")[0] for other, text in sources.items())]
        reports = [r for r in (FOLDER / f"{script.stem}.md", FOLDER / f"{script.stem}.txt") if r.exists()]
        title, detail = docstring_summary(script)
        found.append({"script": script, "charts": mine, "report": reports[0] if reports else None,
                      "title": title, "detail": detail,
                      "updated": max((c.stat().st_mtime for c in mine), default=0)})
    claimed = {c for a in found for c in a["charts"]}
    for chart in charts:  # charts no script mentions still get shown
        if chart not in claimed:
            found.append({"script": None, "charts": [chart], "report": None, "detail": "",
                          "title": chart.stem.replace("_", " ").capitalize(), "updated": chart.stat().st_mtime})
    return sorted(found, key=lambda a: a["updated"], reverse=True)


def repository() -> tuple[str, str] | None:
    """(user, repository) from the git remote, if one is set."""
    remote = git("remote", "get-url", "origin", check=False).stdout.strip()
    match = re.search(r"github\.com[:/]([^/]+)/(.+?)(?:\.git)?/?$", remote)
    return (match.group(1), match.group(2)) if match else None


def build_site(algorithms: list[dict], repo: tuple[str, str] | None) -> int:
    if CHARTS.exists():
        shutil.rmtree(CHARTS)
    CHARTS.mkdir(parents=True)
    (SITE / ".nojekyll").write_text("")  # serve the files as they are
    source = f"https://github.com/{repo[0]}/{repo[1]}" if repo else None

    cards, count = [], 0
    for a in algorithms:
        if not a["charts"]:
            continue
        links = []
        if source and a["script"]:
            links.append(f'<a href="{source}/blob/main/{a["script"].name}">Code: {a["script"].name}</a>')
        elif a["script"]:
            links.append(f"<span>Code: {a['script'].name}</span>")
        if source and a["report"]:
            links.append(f'<a href="{source}/blob/main/{a["report"].name}">Full report</a>')
        images = []
        for chart in a["charts"]:
            shutil.copy2(chart, CHARTS / chart.name)
            count += 1
            images.append(f'<a href="charts/{chart.name}"><img loading="lazy" src="charts/{chart.name}" '
                          f'alt="{html.escape(a["title"])} ({chart.name})"></a>')
        updated = datetime.fromtimestamp(a["updated"]).strftime("%d %B %Y")
        cards.append(f"""
    <article>
      <h2>{html.escape(a["title"])}</h2>
      {f'<p>{html.escape(a["detail"])}</p>' if a["detail"] else ''}
      <p class="meta">Chart updated {updated}{''.join(' · ' + link for link in links)}</p>
      {''.join(images)}
    </article>""")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="{html.escape(TAGLINE)}">
  <title>{TITLE}</title>
  <style>
    :root {{ color-scheme: light dark; --bg: #f6f7f9; --card: #fff; --ink: #1c2330; --soft: #5d6878; --line: #dde2ea; --link: #1a5fb4; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg: #12161d; --card: #1b212b; --ink: #e6eaf0; --soft: #9aa6b8; --line: #2b3442; --link: #7ab3ff; }} }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
    header, main, footer {{ max-width: 1100px; margin: 0 auto; padding: 0 20px; }}
    header {{ padding-top: 48px; padding-bottom: 8px; }}
    h1 {{ margin: 0 0 6px; font-size: 2.2rem; }}
    header p {{ margin: 0 0 14px; color: var(--soft); font-size: 1.1rem; }}
    .note {{ border-left: 4px solid #c9892b; background: var(--card); padding: 10px 14px; border-radius: 4px; font-size: .95rem; color: var(--ink); }}
    article {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 20px; margin: 24px 0; }}
    h2 {{ margin: 0 0 8px; font-size: 1.25rem; }}
    article p {{ margin: 0 0 10px; }}
    .meta {{ color: var(--soft); font-size: .9rem; }}
    a {{ color: var(--link); }}
    img {{ display: block; width: 100%; height: auto; margin-top: 12px; border-radius: 6px; border: 1px solid var(--line); background: #fff; }}
    footer {{ color: var(--soft); font-size: .9rem; padding-top: 8px; padding-bottom: 48px; }}
  </style>
</head>
<body>
  <header>
    <h1>{TITLE}</h1>
    <p>{html.escape(TAGLINE)}</p>
    <p class="note">{html.escape(DISCLAIMER)}</p>
    {f'<p><a href="{source}">All the code is on GitHub</a></p>' if source else ''}
  </header>
  <main>{''.join(cards)}
  </main>
  <footer>
    <p>{count} charts from {sum(1 for a in algorithms if a["script"])} scripts. Site built {datetime.now():%d %B %Y}
    by publish.py. Click a chart to open it full size.</p>
  </footer>
</body>
</html>
"""
    (SITE / "index.html").write_text(page)
    return count


def build_readme(algorithms: list[dict], repo: tuple[str, str] | None) -> None:
    site = f"https://{repo[0]}.github.io/{repo[1]}/" if repo else None
    preview = next((c for a in algorithms for c in a["charts"]), None)
    lines = [f"# {TITLE}", "", TAGLINE, ""]
    if site:
        lines += [f"## See every chart on the website: [{site.removeprefix('https://').rstrip('/')}]({site})", "",
                  f"[![Latest chart]({preview.name})]({site})" if preview else "", ""]
    else:
        lines += ["The charts are collected in a website in `docs/` (open `docs/index.html`).", ""]
    lines += [f"> {DISCLAIMER}", "", "## Algorithms", "", "| Script | What it does | Charts | Report |", "|---|---|---|---|"]
    for a in sorted((a for a in algorithms if a["script"]), key=lambda a: a["script"].name):
        charts = ", ".join(f"[{c.name}]({c.name})" for c in a["charts"]) or "-"
        report = f"[{a['report'].name}]({a['report'].name})" if a["report"] else "-"
        lines.append(f"| [`{a['script'].name}`]({a['script'].name}) | {a['title']} | {charts} | {report} |")
    lines += [
        "", "## Run it yourself", "",
        "```", "python -m venv .venv", ".venv/bin/pip install -r requirements.txt",
        ".venv/bin/python quant_research.py     # or any other script", "```", "",
        "Each script downloads its own data (Yahoo Finance, and GDELT for news tone) and saves a PNG next to itself.",
        "", "## Updating the website", "",
        "`python publish.py` rebuilds the website from every PNG in the folder, rewrites this README, "
        "commits and pushes. New algorithms are picked up automatically.", "",
        "*This README is written by `publish.py`; edits made here by hand are overwritten.*", "",
    ]
    (FOLDER / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the website and upload everything to GitHub.")
    parser.add_argument("--repo", help="GitHub repository as USERNAME/REPOSITORY (needed the first time)")
    parser.add_argument("--no-push", action="store_true", help="build and commit, but do not upload")
    parser.add_argument("--message", "-m", help="commit message")
    args = parser.parse_args()

    if not (FOLDER / ".git").exists():
        git("init", "-b", "main")
        print("Created a git repository in this folder.")
    if args.repo:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
            raise SystemExit("--repo must look like USERNAME/REPOSITORY.")
        url = f"https://github.com/{args.repo}.git"
        has_remote = git("remote", "get-url", "origin", check=False).returncode == 0
        git("remote", "set-url" if has_remote else "add", "origin", url)
    (FOLDER / ".gitignore").write_text(IGNORE)
    if not (FOLDER / "requirements.txt").exists():
        (FOLDER / "requirements.txt").write_text(REQUIREMENTS)

    repo = repository()
    algorithms = find_algorithms()
    count = build_site(algorithms, repo)
    build_readme(algorithms, repo)
    print(f"Website built: {count} charts from {sum(1 for a in algorithms if a['script'])} scripts "
          f"-> {SITE / 'index.html'}")

    if not git("config", "user.name", check=False).stdout.strip() or not git("config", "user.email", check=False).stdout.strip():
        raise SystemExit('\nGit does not know who you are yet, so nothing was committed. Run these once, then run '
                         'publish.py again:\n  git config --global user.name "Your Name"\n'
                         '  git config --global user.email "you@example.com"')
    git("add", "-A")
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("Nothing has changed since the last commit.")
    else:
        git("commit", "-m", args.message or f"Update algorithms and website ({datetime.now():%Y-%m-%d %H:%M})")
        print("Committed: " + git("log", "-1", "--format=%h %s").stdout.strip())

    if args.no_push:
        return
    if not repo:
        raise SystemExit("\nNot uploaded: no GitHub repository is set. Create an empty repository on github.com, "
                         "then run:\n  python publish.py --repo USERNAME/REPOSITORY")
    push = subprocess.run(["git", "push", "-u", "origin", "main"], cwd=FOLDER)  # shows git's own login prompt
    if push.returncode != 0:
        raise SystemExit("\nUpload failed (see git's message above). The commit is saved; run publish.py again "
                         "once it is fixed.")
    print(f"\nUploaded to https://github.com/{repo[0]}/{repo[1]}")
    print(f"Website:  https://{repo[0]}.github.io/{repo[1]}/")
    print("  (first time only: on GitHub open Settings > Pages, choose branch 'main' and folder '/docs', Save)")


if __name__ == "__main__":
    main()
