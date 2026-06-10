import mlflow, json_repair, os
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("http://localhost:5000")
c = MlflowClient()
rid = "b0093c4080ba47c19a51a201f827f188"
subs = {"int_to_roman":"step_2/code_int_to_roman/output.txt",
        "calculate":"step_2/code_evaluate_expression/output.txt",
        "decode_string":"step_2/code_decode_string/output.txt",
        "merge_intervals":"step_2/code_merge_intervals/output.txt"}
def compiles(c0):
    try: compile(c0,"<c>","exec"); return "OK"
    except SyntaxError as e: return f"SyntaxErr L{e.lineno}:{e.msg}"
for name, path in subs.items():
    d = f"/tmp/goal3/raw_{name}"; os.makedirs(d, exist_ok=True)
    p = c.download_artifacts(rid, path, d)
    raw = open(p).read()
    obj = json_repair.loads(raw)
    code = obj["code"] if isinstance(obj, dict) else ""
    print("="*55, name)
    print("  decoded:", compiles(code), "| literal-\\n:", "\\n" in code, "real-nl:", "\n" in code)
    if compiles(code) != "OK" and "\\n" in code:
        rep = code.encode("utf-8").decode("unicode_escape")
        print("  unicode_escape repair:", compiles(rep))
