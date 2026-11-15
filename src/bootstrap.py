"""Colab bootstrap. Run from a notebook cell with SESSION = "a" or "b" defined:

    installs dependencies, pulls the code from the private ops repo, runs a smoke test of the whole
    path in an isolated namespace, and only if that passes runs the real session. Logs are uploaded
    to the ops repo every few minutes so the run can be monitored remotely.
"""
import os
import subprocess
import sys
import threading
import time

from huggingface_hub import HfApi, snapshot_download, whoami

SESSION = globals().get("SESSION", "a")
USER = whoami()["name"]
OPS = f"{USER}/xlrb-ops"
LOG = f"/content/session_{SESSION}.log"
PIP = {"a": ['"transformers>=5.15,<6"', "sacrebleu", "sentencepiece", "safetensors", "pyyaml", "curatorkit"],
       "b": ["safetune", "circuitkit", '"auditkit[transformers]"', "sacrebleu", "sentencepiece", "safetensors", "pyyaml"]}


def sh(cmd, env=None):
    with open(LOG, "a") as log:
        log.write(f"\n$ {cmd}\n")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env={**os.environ, **(env or {})}, cwd="/content")
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return p.wait()


def upload_log():
    try:
        HfApi().upload_file(path_or_fileobj=LOG, path_in_repo=f"logs/session_{SESSION}.log", repo_id=OPS,
                            repo_type="dataset", commit_message="chore: update session log")
    except Exception as e:  # noqa: BLE001
        print("[bootstrap] log upload failed:", e)


def heartbeat():
    while True:
        time.sleep(240)
        upload_log()


open(LOG, "a").write(f"[bootstrap] session {SESSION} start {time.strftime('%FT%TZ', time.gmtime())}\n")
threading.Thread(target=heartbeat, daemon=True).start()
if SESSION == "b":
    sh("pip -q uninstall -y torchao")
sh("pip -q install " + " ".join(PIP[SESSION]))
code = snapshot_download(OPS, repo_type="dataset", allow_patterns=["code/*"], local_dir="/content/xlrb_code")
env = {"XLRB_HF_USER": USER, "PYTHONPATH": "/content/xlrb_code/code/src",
       "XLRB_CODE_SHA": open("/content/xlrb_code/code/CODE_SHA").read().strip(), "PYTHONUNBUFFERED": "1"}
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
ns = "quick_" + env["XLRB_CODE_SHA"]
rc = sh(f"python -u -m xlrb.cli run_{SESSION} --quick", {**env, "XLRB_NS": ns, "XLRB_WORK": "/content/xlrb_quick"})
open(LOG, "a").write(f"[bootstrap] smoke test exit code {rc}\n")
upload_log()
if rc == 0:
    rc = sh(f"python -u -m xlrb.cli run_{SESSION}", {**env, "XLRB_WORK": "/content/xlrb_work"})
    open(LOG, "a").write(f"[bootstrap] full run exit code {rc}\n")
else:
    print("[bootstrap] smoke test failed; full run not started")
upload_log()
