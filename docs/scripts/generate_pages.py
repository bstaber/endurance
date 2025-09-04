"""Generate API reference documentation for the Kernax package."""

from pathlib import Path

import mkdocs_gen_files
from mkdocs_gen_files.nav import Nav

PKG_NAME = "kernax"
SRC_ROOT = Path("src") / PKG_NAME
OUT_DIR = Path("reference")

nav = Nav()

for path in sorted(SRC_ROOT.rglob("*.py")):
    rel_mod = path.relative_to("src").with_suffix("")
    parts = tuple(rel_mod.parts)

    if parts[-1] == "__main__":
        continue

    doc_path = Path(*parts).with_suffix(".md")
    full_doc_path = OUT_DIR / doc_path

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = Path(*parts) / "index.md"
        full_doc_path = OUT_DIR / doc_path

    if parts[-1] in ("types", "toy_mixtures", "__init__"):
        continue

    # record in nav (relative to OUT_DIR)
    nav[parts] = full_doc_path.as_posix()

    # write the stub
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts) if parts else PKG_NAME
        fd.write(f"# {identifier.lower()}\n\n")
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# top-level API landing page
# with mkdocs_gen_files.open(OUT_DIR / "index.md", "w") as fd:
#     fd.write("# API reference\n")

# nav file for literate-nav
with mkdocs_gen_files.open("SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
