import json_repair
from pathlib import Path
def compiles(c):
    try:
        compile(c, "<c>", "exec"); return True
    except SyntaxError as e:
        return f"SyntaxError L{e.lineno}: {e.msg}"
raws = {p.parent.name: (p.read_text()) for p in Path("/tmp/goal3/raw").rglob("output.txt")}
# fallback: re-read from the four known files
import glob
for f in sorted(glob.glob("/tmp/goal3/raw/**/output.txt", recursive=True)):
    name = f.split("/")[-2]
    raw = open(f).read()
    code = json_repair.loads(raw)["code"] if isinstance(json_repair.loads(raw), dict) else ""
    has_lit = "\\n" in code and "\n" not in code.split("\\n")[0]
    print("="*60); print(name)
    print("  decoded compiles?:", compiles(code))
    print("  has literal backslash-n:", "\\n" in code, "| has real newline:", "\n" in code)
    if "\\n" in code and compiles(code) is not True:
        repaired = code.encode("utf-8").decode("unicode_escape")
        print("  -> unicode_escape repair compiles?:", compiles(repaired))
