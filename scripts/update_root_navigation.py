from __future__ import annotations

from pathlib import Path

import pymdownx.superfences  # type: ignore[import-untyped]  # noqa: F401  (needed for PyYAML !!python/name constructors)
import yaml


def update_navigation():
    """
    Dynamically updates the 'Components' section in mkdocs.yml based on
    subdirectories in services/, libs/, and apps/ that contain their own 'mkdocs.yml'.
    """
    mkdocs_file = Path("mkdocs.yml")
    if not mkdocs_file.exists():
        print("Error: mkdocs.yml not found.")
        return

    # Use FullLoader so `mkdocs.yml` can include tags like
    # `!!python/name:pymdownx.superfences.fence_div_format` (used by pymdownx).
    loader = getattr(yaml, "FullLoader", yaml.Loader)
    dumper = getattr(yaml, "Dumper", yaml.SafeDumper)

    with open(mkdocs_file, "r") as f:
        try:
            config = yaml.load(f, Loader=loader)
        except yaml.YAMLError as e:
            print(f"Error parsing mkdocs.yml: {e}")
            return

    if not config or "nav" not in config:
        print("Error: 'nav' section not found in mkdocs.yml")
        return

    nav = config["nav"]

    # Rebuild the Components list
    new_components_list = []

    # Always put Overview first
    new_components_list.append({"Overview": "components-overview.md"})

    # Scan services/, libs/, apps/ for sub-projects (each with mkdocs.yml)
    root_dirs = [
        ("services", Path("services")),
        ("libs", Path("libs")),
        ("apps", Path("apps")),
    ]
    found = []
    for label, dir_path in root_dirs:
        if dir_path.exists():
            for item in dir_path.iterdir():
                if item.is_dir() and (item / "mkdocs.yml").exists():
                    found.append((item.name, f"./{label}/{item.name}/mkdocs.yml"))

    for name, include_path in sorted(found, key=lambda x: x[0]):
        # NOTE: Use a quoted string, not a YAML tag, so MkDocs can parse
        # the config without custom YAML constructors.
        new_components_list.append({name: f"!include {include_path}"})

    # Find and update (or add) the "Components" section
    components_idx = -1
    for i, entry in enumerate(nav):
        if isinstance(entry, dict) and "Components" in entry:
            components_idx = i
            break

    if components_idx != -1:
        # Update existing
        nav[components_idx]["Components"] = new_components_list
    else:
        # Append new
        nav.append({"Components": new_components_list})

    # Save back to mkdocs.yml
    with open(mkdocs_file, "w") as f:
        # sort_keys=False preserves top-level order
        # default_flow_style=False ensures block format (nice nesting)
        yaml.dump(
            config,
            f,
            Dumper=dumper,
            sort_keys=False,
            default_flow_style=False,
            indent=2,
        )

    print(
        f"Successfully updated mkdocs.yml navigation with {len(new_components_list) - 1} component entries."
    )


if __name__ == "__main__":
    update_navigation()
