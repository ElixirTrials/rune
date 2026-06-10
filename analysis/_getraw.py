import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("http://localhost:5000")
c = MlflowClient()
run_id = "b0093c4080ba47c19a51a201f827f188"
# list artifacts under step_2
for sub in ["step_2/code_int_to_roman","step_2/code_evaluate_expression","step_2/code_decode_string","step_2/code_merge_intervals"]:
    try:
        arts = c.list_artifacts(run_id, sub)
        for a in arts:
            if a.path.endswith("output.txt"):
                p = c.download_artifacts(run_id, a.path, "/tmp/goal3/raw")
                raw = open(p).read()
                print("="*70); print(sub)
                print("repr(first 200):", repr(raw[:200]))
    except Exception as e:
        print(sub, "ERR", e)
