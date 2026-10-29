"""Local coding-agent iterations with a fixed remote GPU evaluator.

The generator edits two runtime files. The evaluator is copied from the
trusted starting snapshot, never from the generator's workspace.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time

from .evaluate import write_json
from .task import load_task

RUNTIME_FILES = ("engine.py", "server.py")


def invoke_generator(command, candidate, prompt, logs, timeout):
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    (logs/"prompt.txt").write_text(prompt)
    started = time.monotonic()
    timed_out = False
    with (logs/"events.jsonl").open("w") as stdout, (logs/"stderr.log").open("w") as stderr:
        process = subprocess.Popen(command,cwd=candidate,stdin=subprocess.PIPE,stdout=stdout,
                                   stderr=stderr,text=True,start_new_session=True)
        try:
            process.communicate(prompt,timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        finally:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
    usage = {}
    completed = False
    for line in (logs/"events.jsonl").read_text().splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if item.get("type") == "turn.completed":
            completed = True
            for key,value in item.get("usage",{}).items():
                if isinstance(value,(float,int)):
                    usage[key] = usage.get(key,0)+value
    result = {"returncode":process.returncode,"timed_out":timed_out,
              "turn_completed":completed,"wall_seconds":time.monotonic()-started,
              "usage":usage or None,"command":command}
    write_json(logs/"result.json",result)
    return result


def materialize(trusted, candidate, bundle):
    trusted,candidate,bundle = map(Path,(trusted,candidate,bundle))
    shutil.copytree(trusted/"forge",bundle/"forge",ignore=shutil.ignore_patterns("__pycache__"))
    for name in RUNTIME_FILES:
        shutil.copyfile(candidate/"forge"/name,bundle/"forge"/name)


class RemoteEvaluator:
    def __init__(self, host, port, key, root, python):
        self.host,self.root,self.python = host,root.rstrip('/'),python
        self.ssh = ["ssh","-i",str(Path(key).expanduser()),"-p",str(port),
                    "-o","BatchMode=yes","-o","ConnectTimeout=15"]

    def call(self, script):
        result = subprocess.run(self.ssh+[self.host,"python3 -"],input=script,text=True,
                                capture_output=True,timeout=45)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result.stdout

    def evaluate(self, bundle, task, destination, name):
        remote = self.root+"/campaigns/"+name
        session = "forge-"+name
        self.call(f"from pathlib import Path\np=Path({remote!r}); p.mkdir(parents=True,exist_ok=False)\n")
        write_json(Path(bundle)/"task.json",task)
        subprocess.run(["rsync","-az","-e",shlex.join(self.ssh),str(bundle)+"/",
                        self.host+":"+remote+"/"],check=True,timeout=90)
        command = [self.python,"-m","forge.pilot","--task","task.json","--output","result"]
        shell = ("cd "+shlex.quote(remote)+" && env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="+
                 shlex.quote(self.root+"/.tools:.")+" "+shlex.join(command)+
                 " > evaluation.log 2>&1; forge_rc=$?; printf '%s\\n' \"$forge_rc\" > exit.code")
        launch = f"import subprocess\nsubprocess.run(['tmux','new-session','-d','-s',{session!r},{shell!r}],check=True)\n"
        self.call(launch)
        destination = Path(destination)
        destination.mkdir(parents=True,exist_ok=True)
        write_json(destination/"remote-job.json",{"host":self.host,"directory":remote,"session":session})
        deadline = time.monotonic()+task["timeout_seconds"]+90
        while True:
            # Poll this exact job. A transport failure does not restart it.
            try:
                status = self.call(f"from pathlib import Path\np=Path({remote!r})/'exit.code'\nprint(p.read_text().strip() if p.exists() else 'RUNNING')\n").strip()
            except (RuntimeError,subprocess.TimeoutExpired) as exc:
                with (destination/"poll-errors.log").open("a") as log:
                    log.write(str(exc)+"\n")
                status = "RUNNING"
            if status != "RUNNING":
                break
            if time.monotonic()>deadline:
                raise TimeoutError(f"Observation timed out; inspect existing {session} at {remote}, do not restart")
            time.sleep(3)
        # Keep stdout even when a stage fails before writing a summary.
        write_json(destination/"remote-exit.json",{"returncode":int(status)})
        try:
            output = self.call(f"from pathlib import Path\nprint((Path({remote!r})/'evaluation.log').read_text())\n")
        except (RuntimeError,subprocess.TimeoutExpired) as exc:
            output = "Log collection failed: "+str(exc)
        (destination/"evaluation.log").write_text(output)
        collection=subprocess.run(["rsync","-az","-e",shlex.join(self.ssh),self.host+":"+remote+"/result/",
                        str(destination)+"/"],capture_output=True,text=True,timeout=90)
        if collection.returncode:
            summary={"status":"blocked","stage":"result_collection","error":collection.stderr,
                     "remote_exit_code":int(status)}
            write_json(destination/"collection-error.json",summary)
            return summary
        summary_path = destination/"summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"status":"blocked"}
        summary["remote_exit_code"] = int(status)
        if int(status)!=0 and summary.get("status")=="pass":
            summary["status"] = "fail"
        return summary


def source_digest(path):
    return {name:hashlib.sha256((Path(path)/"forge"/name).read_bytes()).hexdigest() for name in RUNTIME_FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--host",required=True)
    parser.add_argument("--port",type=int,default=22)
    parser.add_argument("--key",required=True)
    parser.add_argument("--remote-root",required=True)
    parser.add_argument("--remote-python",required=True)
    parser.add_argument("--rounds",type=int,default=1)
    parser.add_argument("--generator-timeout",type=int,default=360)
    parser.add_argument("--generator-model",default=None)
    parser.add_argument("--clean-generator-config",action="store_true",help="Use CLI defaults plus saved auth, without optional user MCP configuration")
    args = parser.parse_args()
    root=Path.cwd();out=Path(args.output).resolve()
    if not re.fullmatch(r'[a-z0-9-]{1,40}',out.name):
        parser.error("output directory name must be short lowercase letters, digits or hyphens")
    out.mkdir(parents=True,exist_ok=False)
    task=load_task(args.task)
    task["benchmark_repeats"]=3
    write_json(out/"task.json",task)
    write_json(out/"manifest.json",{"started_unix":time.time(),"round_limit":args.rounds,
               "generator_timeout_seconds":args.generator_timeout,
               "task_sha256":hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest(),
               "base_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
               "codex_version":subprocess.check_output(["codex","--version"],text=True).strip(),
               "generator_model":args.generator_model or "user-configured default; raw execution events retained",
               "clean_generator_config":args.clean_generator_config,
               "start_mode":"seeded complete runtime"})
    trusted=out/"trusted"
    shutil.copytree(root/"forge",trusted/"forge",ignore=shutil.ignore_patterns("__pycache__"))
    baseline=out/"seed"
    (baseline/"forge").mkdir(parents=True)
    for name in RUNTIME_FILES:
        shutil.copyfile(trusted/"forge"/name,baseline/"forge"/name)
    remote=RemoteEvaluator(args.host,args.port,args.key,args.remote_root,args.remote_python)
    history=[];best=baseline;best_score=None
    for index in range(args.rounds+1):
        rd=out/f"round-{index:02d}";candidate=rd/"candidate"
        (candidate/"forge").mkdir(parents=True)
        for name in RUNTIME_FILES:
            shutil.copyfile(best/"forge"/name,candidate/"forge"/name)
        generation=None
        if index:
            prompt=("Optimize the complete Qwen2 inference runtime in forge/engine.py and forge/server.py. "
                    "Edit only those two files. This is a seeded construction iteration. Preserve public interfaces, "
                    "float32, disabled TF32, true model computation, per-request KV, token/text correctness and token-round-robin service. "
                    "Do not use Transformers model/generate or substitute another serving engine. Torch primitives are allowed. "
                    "Target one A40, Qwen2.5-0.5B, four concurrent requests of 16 output tokens, max context512. "
                    "Reduce actual serving time. Do not hardcode outputs or change validation. Do not run tests or inference locally; "
                    "the controller runs the unchanged evaluator on the remote GPU after you finish. No SSH, credentials, Git operations "
                    "or unrelated files. Choose a concrete implementation improvement; explain what changed and why. "
                    "Previous measured feedback:\n"+json.dumps(history,ensure_ascii=False))
            command=["codex","exec","--ephemeral","--json","--skip-git-repo-check","--sandbox","workspace-write",
                     "-C",str(candidate),"-"]
            if args.generator_model:
                command[2:2] = ["--model",args.generator_model]
            if args.clean_generator_config:
                command.insert(2,"--ignore-user-config")
            generation=invoke_generator(command,candidate,prompt,rd/"generator",args.generator_timeout)
            if generation["returncode"]!=0 or not generation["turn_completed"]:
                history.append({"round":index,"status":"generator_failed","generation":generation})
                write_json(out/"history.json",history)
                continue
            if source_digest(candidate)==source_digest(best):
                history.append({"round":index,"status":"no_code_change","generation":generation})
                write_json(out/"history.json",history)
                continue
        bundle=rd/"bundle";materialize(trusted,candidate,bundle)
        digest=source_digest(bundle)
        try:
            summary=remote.evaluate(bundle,task,rd/"evaluation",out.name+f"-r{index}")
        except Exception as exc:
            history.append({"round":index,"status":"blocked","error":str(exc),"evaluation":str(rd/"evaluation")})
            write_json(out/"history.json",history)
            break
        provenance_path=rd/"evaluation/provenance.json"
        if summary.get("status")=="pass":
            measured=json.loads(provenance_path.read_text())["source"]["files"] if provenance_path.exists() else {}
            if any(measured.get("forge/"+name)!=digest[name] for name in RUNTIME_FILES):
                summary={"status":"fail","error":"measured source identity mismatch"}
        bench=summary.get("benchmark") or {}
        score=bench.get("median_tokens_per_second") if summary.get("status")=="pass" else None
        entry={"round":index,"status":summary.get("status"),"score_tokens_per_second":score,
               "source":digest,"generation":generation,"evaluation":str(rd/"evaluation")}
        if summary.get("error"):
            entry["error"]=summary["error"]
        for fname in ["numerics.json","blocked.json"]:
            path=rd/"evaluation"/fname
            if path.exists() and summary.get("status")!="pass":
                entry["failure_detail"]=path.read_text()[:8000]
        history.append(entry)
        if score is not None and (best_score is None or score>best_score):
            best,best_score=bundle,score
        write_json(out/"history.json",history)
        print(json.dumps({"round":index,"status":entry["status"],"score":score}),flush=True)
    generated_valid=any(r["round"]>0 and r["status"]=="pass" for r in history)
    write_json(out/"result.json",{"status":"generated_and_evaluated" if generated_valid else "no_valid_generated_candidate","best_candidate":str(best),"best_measured_score":best_score,
               "rounds":history,"scope":"seeded development optimization; not from-scratch synthesis or statistically established advantage"})
    raise SystemExit(0 if generated_valid else 1)


if __name__=="__main__":
    main()
