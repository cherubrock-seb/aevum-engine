#!/usr/bin/env python3
"""Kaggle T4 short CUDA/OpenCL PFA7 carryB A/B. No production changes."""
import argparse, ctypes, glob, json, os, pathlib, re, shutil, subprocess, sys, time, zipfile

SRC = pathlib.Path(__file__).resolve().parent.parent
ROOT = pathlib.Path("/kaggle/working") if pathlib.Path("/kaggle/working").exists() else pathlib.Path.cwd()
WORK = ROOT / "pfa7_carryb_ab"
LOGS = WORK / "logs"
LINK = WORK / "cuda_link"
LOGS.mkdir(parents=True, exist_ok=True)
LINK.mkdir(parents=True, exist_ok=True)
CASES = [("classic","0","0"),("lead","1","0"),("carry-index","1","1"),("carry-early","1","2")]

def call(name, argv, env=None, timeout=900):
    start=time.monotonic()
    with (LOGS/name).open("w",encoding="utf8",errors="replace") as f:
        try:
            rc=subprocess.run(argv,cwd=SRC,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
        except Exception as e:
            rc=124 if isinstance(e,subprocess.TimeoutExpired) else 125
            f.write("\nERROR="+repr(e)+"\n")
    print(name,"rc=",rc,"s=",round(time.monotonic()-start,1),flush=True)
    if rc: print("\n".join((LOGS/name).read_text(errors="replace").splitlines()[-12:]),flush=True)
    return rc

def cuda_driver():
    paths=[]
    try:
        out=subprocess.run(["ldconfig","-p"],capture_output=True,text=True).stdout
        paths.extend(s.split("=>",1)[1].strip() for s in out.splitlines() if "libcuda.so.1" in s and "=>" in s)
    except Exception: pass
    for d in ["/usr/lib/x86_64-linux-gnu","/usr/local/nvidia/lib64","/usr/lib64","/lib/x86_64-linux-gnu"]:
        paths += glob.glob(d+"/libcuda.so.1")
    for item in paths:
        p=pathlib.Path(item)
        if p.exists() and "stubs" not in str(p):
            try: ctypes.CDLL(str(p)); return p.resolve()
            except OSError: pass
    return None

def run(args):
    print("SOURCE_SHA=",subprocess.check_output(["git","rev-parse","HEAD"],cwd=SRC,text=True).strip(),flush=True)
    subprocess.run(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],check=False)
    for test in ["tests/pfa7_lead_mode_source_test.py","tests/pfa7_carryb_predecessor_model.py","tests/pfa7_prpll_native_layout_map_model.py"]:
        if call("gate_"+pathlib.Path(test).stem+".log",[sys.executable,test],timeout=120):
            raise RuntimeError("Host gate failed: "+test)
    driver=cuda_driver()
    print("CUDA_DRIVER=",driver)
    if driver:
        link=LINK/"libcuda.so"
        if link.exists() or link.is_symlink(): link.unlink()
        link.symlink_to(driver)
    if call("bundle.log",["make","-B","src/bundle.cpp"],timeout=240):
        raise RuntimeError("kernel bundle failed")
    libs={}
    for backend in ("cuda","opencl"):
        if backend=="cuda" and (args.skip_cuda or not driver): continue
        if backend=="opencl" and args.skip_opencl: continue
        call(backend+"_clean.log",["make","clean"],timeout=120)
        cmd=["make","-j4","engine-lib"]
        if backend=="cuda": cmd+=["CUDA=1","LIBPATH=-L"+str(LINK)]
        rc=call(backend+"_build.log",cmd,timeout=1200)
        lib=SRC/"build-engine/libaevum_engine.so"
        if rc==0 and lib.is_file():
            dest=WORK/("lib_"+backend+".so")
            shutil.copy2(lib,dest)
            libs[backend]=dest
        else: print("BACKEND_UNAVAILABLE",backend,flush=True)
    env0=os.environ.copy()
    env0["LD_LIBRARY_PATH"]=":".join([str(LINK),"/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",env0.get("LD_LIBRARY_PATH","")])
    results=[];statuses=[];type2={}
    for backend,lib in libs.items():
        for label,lead,carry in CASES:
            env=env0.copy()
            env.update({"AEVUM_PFA7_LEAD_MODE":lead,"AEVUM_PFA7_CARRYB_FAST":carry,
                "AEVUM_EXPERIMENT_FFT3261":"1","AEVUM_AUTOTUNE":"off"})
            for key in ["AEVUM_PFA_USE","AEVUM_PFA7_INPLACE","AEVUM_PFA_GF31_COSBASE",
                        "AEVUM_PFA_LEAD_BRIDGE"]:
                env.pop(key,None)
            logfile="probe_"+backend+"_"+label+".log"
            rc=call(logfile,[sys.executable,str(SRC/"tools/pfa7_kaggle_probe.py"),
                "--lib",str(lib),"--kernels",str(SRC),"--device",str(args.device),
                "--exponent","136279841","--iterations",str(args.iterations),
                "--mode",lead],env=env,timeout=600)
            log=(LOGS/logfile).read_text(errors="replace")
            exact=re.findall(r"^PFA7_WORD_EXACT=(PASS|FAIL)",log,re.M)
            perf=re.findall(r"^PFA7_RESULT=(\{.*\})$",log,re.M)
            control=re.findall(r"^TYPE2_REFERENCE=(\{.*\})$",log,re.M)
            if control: type2[backend]=json.loads(control[-1])
            status={"backend":backend,"case":label,"rc":rc,
                    "exact":exact[-1] if exact else "NOT_RUN","log":logfile}
            statuses.append(status)
            if rc==0 and exact and exact[-1]=="PASS" and perf:
                result=json.loads(perf[-1])
                result.update({"case":label,"backend":backend})
                results.append(result)
                print("EXACT_PASS",backend,label,result["iter_s"],"iter/s",flush=True)
            else: print("NOT_VALIDATED",status,flush=True)
    return libs,results,statuses,type2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--iterations",type=int,default=192)
    ap.add_argument("--device",type=int,default=0)
    ap.add_argument("--skip-cuda",action="store_true")
    ap.add_argument("--skip-opencl",action="store_true")
    args=ap.parse_args()
    libs={};results=[];statuses=[];type2={};error=None
    try: libs,results,statuses,type2=run(args)
    except Exception as exc:
        error=repr(exc)
        print("FATAL=",error,flush=True)
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=SRC,capture_output=True,text=True).stdout.strip()
    (WORK/"manifest.json").write_text(json.dumps({
        "sha":head,"exponent":136279841,"iterations":args.iterations,
        "configurations":CASES,"gpu_device":args.device,"backends":list(libs),
        "type2":type2,"status":statuses,"results":results,"error":error},indent=2))
    artifact=ROOT/"pfa7-carryb-kaggle-evidence.zip"
    with zipfile.ZipFile(artifact,"w",zipfile.ZIP_DEFLATED) as z:
        for log in sorted(LOGS.glob("*.log")): z.write(log,"logs/"+log.name)
        z.write(WORK/"manifest.json","manifest.json")
    print("EVIDENCE_ZIP=",artifact,flush=True)
    for backend in libs:
        base=next((v for v in results if v["backend"]==backend and v["case"]=="lead"),None)
        for label in ["carry-index","carry-early"]:
            test=next((v for v in results if v["backend"]==backend and v["case"]==label),None)
            if base and test:
                delta=100*(test["iter_s"]/base["iter_s"]-1)
                print(backend.upper(),label,"vs same-plan lead:",f"{delta:+.2f}%",flush=True)
    return 0 if statuses and all(x["rc"]==0 and x["exact"]=="PASS" for x in statuses) else 1

if __name__=="__main__":
    sys.exit(main())
