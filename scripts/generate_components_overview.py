from pathlib import Path


def generate_overview():
    """
    Scans services/, libs/, and apps/ and generates a markdown overview page
    with links to each component's documentation.
    """
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    overview_file = docs_dir / "components-overview.md"

    content = [
        "# Components Overview",
        "",
        "This page provides an overview of all microservices and shared libraries in this monorepo.",
        "",
        "## Available Components",
        "",
        "| Component | Description | Documentation |",
        "| :--- | :--- | :--- |",
    ]

    root_dirs = [Path("services"), Path("libs"), Path("apps")]
    found_any = False
    items_with_path = []
    for parent in root_dirs:
        if not parent.exists():
            continue
        for item in parent.iterdir():
            if item.is_dir() and (item / "mkdocs.yml").exists():
                items_with_path.append(item)

    for item in sorted(items_with_path, key=lambda p: p.name):
        found_any = True
        description = "No description available."
        readme = item / "README.md"

        if readme.exists():
            with open(readme, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.startswith("## Purpose") or line.startswith(
                        "## Description"
                    ):
                        for j in range(i + 1, min(i + 4, len(lines))):
                            if lines[j].strip():
                                description = lines[j].strip()
                                break
                        break
                    if not description or description == "No description available.":
                        if line.strip() and not line.startswith("#"):
                            description = line.strip()

        # Link to component's API reference (each sub-project has docs_dir: docs, nav: api/index.md)
        link = f"{item.name}/api/index.md"
        content.append(f"| **{item.name}** | {description} | [API Reference]({link}) |")

    if not found_any:
        content.append("| - | No components found | - |")

    with open(overview_file, "w") as f:
        f.write("\n".join(content) + "\n")

    print(f"Generated {overview_file}")


if __name__ == "__main__":
    generate_overview()
