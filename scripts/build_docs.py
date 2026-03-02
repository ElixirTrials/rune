import yaml
import sys
from mkdocs.__main__ import cli


# Define and register the !include constructor to avoid SafeLoader errors
# This mimics what the plugin does, but ensures it happens before MkDocs parses the config
def include_constructor(loader, node):
    return f"!include {loader.construct_scalar(node)}"


# Register on both SafeLoader and Loader to be safe
yaml.SafeLoader.add_constructor("!include", include_constructor)
if hasattr(yaml, "Loader"):
    yaml.Loader.add_constructor("!include", include_constructor)
# Be aggressive: if CSafeLoader exists (LibYAML), patch it too
if hasattr(yaml, "CSafeLoader"):
    yaml.CSafeLoader.add_constructor("!include", include_constructor)

if __name__ == "__main__":
    # Pass control to MkDocs CLI
    sys.exit(cli())
