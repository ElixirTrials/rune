import yaml
from pathlib import Path


# 1. Define a class to represent the tagged value
class IncludeTag(str):
    pass


# 2. Define a representer for this class
def include_representer(dumper, data):
    # represent_scalar(tag, value)
    return dumper.represent_scalar("!include", str(data))


# 3. Define a constructor (optional, if you want to read existing tags)
def include_constructor(loader, node):
    return IncludeTag(loader.construct_scalar(node))


# 4. Register them
yaml.SafeLoader.add_constructor("!include", include_constructor)
yaml.SafeDumper.add_representer(IncludeTag, include_representer)


def update_navigation():
    """
    Dynamically updates the 'Components' section in mkdocs.yml based on
    subdirectories in services/, libs/, and apps/ that contain their own 'mkdocs.yml'.
    """
    mkdocs_file = Path("mkdocs.yml")
    if not mkdocs_file.exists():
        print("Error: mkdocs.yml not found.")
        return

    with open(mkdocs_file, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.SafeLoader)
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
        tag_object = IncludeTag(include_path)
        new_components_list.append({name: tag_object})

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
            Dumper=yaml.SafeDumper,
            sort_keys=False,
            default_flow_style=False,
            indent=2,
        )

    print(
        f"Successfully updated mkdocs.yml navigation with {len(new_components_list) - 1} component entries."
    )


if __name__ == "__main__":
    update_navigation()
