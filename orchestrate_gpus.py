#!/usr/bin/env python3
"""
Dynamic multi-GPU orchestrator for process_pisco_profiles.py (Option A).

Instead of statically assigning whole profiles to GPUs up front, this keeps a
shared work queue of profiles and runs one persistent worker per GPU. Each
worker pulls the next profile whenever it becomes free and processes it by
invoking process_pisco_profiles.py as a subprocess pinned to that GPU
(HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES) with a capped detection-core count
(PISCO_N_CORES). This removes the idle-GPU imbalance that happens when profile
sizes differ a lot: no GPU sits idle while another grinds through a big profile.

Profiles are ordered largest-first (by image count) so the long jobs start
early — the classic Longest-Processing-Time greedy that minimizes makespan on
two machines. Disable with --no-lpt.

Nothing in the core pipeline changes: each profile is still processed by the
existing single-profile code path, so the normal single/split workflows keep
working unchanged. Concurrency is safe because distinct profiles write to
distinct output dirs and distinct Postgres rows.

Examples:
  # Two XTX cards, 8 detection cores each, all M181 profiles, LPT-ordered:
  HIP_env is set per-worker automatically; just run with the pisco env python:
  python orchestrate_gpus.py --cruise M181 \
      --output /media/veit/T710_data/pisco_processed/M181 \
      --gpus 0,1 --n-cores-per-gpu 8

  # Explicit subset from a file, and forward extra flags to the pipeline:
  python orchestrate_gpus.py --cruise M181 --output /path/out \
      --profiles-file ./m181_selection.txt --gpus 0,1 -- --export-zip --config ./cfg.json
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESS_SCRIPT = os.path.join(HERE, "process_pisco_profiles.py")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_profiles_file(path):
    profiles = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                profiles.append(line)
    return profiles


def discover_profiles(source, cruise):
    base = os.path.join(source, cruise, f"{cruise}-PISCO-Profiles")
    if not os.path.isdir(base):
        log(f"ERROR: PISCO-Profiles directory not found: {base}")
        return [], base
    profiles = sorted(
        d for d in os.listdir(base)
        if not d.startswith(".") and os.path.isdir(os.path.join(base, d))
    )
    return profiles, base


def estimate_size(profile_dir, image_ext):
    """Cheap image-count proxy: max image-file count in the profile dir or any
    of its immediate subdirs (matches the common <profile>/PNG layout)."""
    ext = image_ext.lower()
    best = 0
    try:
        entries = list(os.scandir(profile_dir))
    except OSError:
        return 0
    best = max(best, sum(1 for e in entries if e.is_file() and e.name.lower().endswith(ext)))
    for e in entries:
        if e.is_dir():
            try:
                c = sum(1 for f in os.scandir(e.path) if f.is_file() and f.name.lower().endswith(ext))
                best = max(best, c)
            except OSError:
                pass
    return best


class Orchestrator:
    def __init__(self, args, passthrough):
        self.args = args
        self.passthrough = passthrough
        self.lock = threading.Lock()
        self.results = []          # list of dicts: profile, gpu, rc, seconds
        self.running_procs = {}    # gpu -> Popen (for clean shutdown)
        self.stop_flag = threading.Event()

    def worker(self, gpu, queue, log_dir):
        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PISCO_N_CORES"] = str(self.args.n_cores_per_gpu)

        while not self.stop_flag.is_set():
            with self.lock:
                if not queue:
                    return
                profile = queue.popleft()

            cmd = [
                sys.executable, PROCESS_SCRIPT,
                "--mode", "cruise",
                "--cruise", self.args.cruise,
                "--source", self.args.source,
                "--output", self.args.output,
                "--profiles", profile,
            ] + self.passthrough

            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in profile)
            log_path = os.path.join(log_dir, f"gpu{gpu}_{safe}.log")
            log(f"GPU{gpu}  START  {profile}")
            t0 = time.time()
            with open(log_path, "w") as lf:
                lf.write(f"# cmd: {' '.join(cmd)}\n")
                lf.write(f"# HIP_VISIBLE_DEVICES={gpu} PISCO_N_CORES={self.args.n_cores_per_gpu}\n\n")
                lf.flush()
                proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
                with self.lock:
                    self.running_procs[gpu] = proc
                rc = proc.wait()
                with self.lock:
                    self.running_procs.pop(gpu, None)
            dt = time.time() - t0
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            log(f"GPU{gpu}  {status:9s} {profile}  ({dt/60:.1f} min)  -> {log_path}")
            with self.lock:
                self.results.append({"profile": profile, "gpu": gpu, "rc": rc, "seconds": dt})

    def run(self, profiles, log_dir):
        queue = deque(profiles)
        threads = []
        for gpu in self.args.gpu_list:
            t = threading.Thread(target=self.worker, args=(gpu, queue, log_dir), daemon=True)
            t.start()
            threads.append(t)
        try:
            for t in threads:
                while t.is_alive():
                    t.join(timeout=1.0)
        except KeyboardInterrupt:
            log("Interrupted — terminating running subprocesses...")
            self.stop_flag.set()
            with self.lock:
                for proc in self.running_procs.values():
                    proc.terminate()
            for t in threads:
                t.join(timeout=10)
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic multi-GPU profile scheduler for process_pisco_profiles.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Anything after a literal '--' is forwarded verbatim to process_pisco_profiles.py.",
    )
    parser.add_argument("--cruise", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="/mnt/filer",
                        help="Root dir containing cruise folders (default: /mnt/filer)")
    parser.add_argument("--profiles-file", help="Newline-delimited profile names (# comments ok)")
    parser.add_argument("--profiles", nargs="+", help="Explicit profile names")
    parser.add_argument("--gpus", default="0,1",
                        help="Comma-separated GPU indices, one worker each (default: 0,1)")
    parser.add_argument("--n-cores-per-gpu", type=int, default=8,
                        help="PISCO_N_CORES per worker; keep sum <= physical cores (default: 8)")
    parser.add_argument("--image-ext", default=".png", help="Image extension for size estimate")
    parser.add_argument("--no-lpt", action="store_true",
                        help="Do not reorder profiles largest-first")
    args, passthrough = parser.parse_known_args()

    # '--' separator: argparse leaves it in passthrough; drop a leading one.
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    args.gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if len(args.gpu_list) < 1:
        parser.error("--gpus must list at least one GPU index")

    # Resolve profile list
    if args.profiles_file:
        profiles = load_profiles_file(args.profiles_file)
    elif args.profiles:
        profiles = args.profiles
    else:
        profiles, _ = discover_profiles(args.source, args.cruise)

    if not profiles:
        log("No profiles to process. Exiting.")
        return 1

    # LPT ordering by image count
    base = os.path.join(args.source, args.cruise, f"{args.cruise}-PISCO-Profiles")
    if not args.no_lpt:
        sized = [(p, estimate_size(os.path.join(base, p), args.image_ext)) for p in profiles]
        sized.sort(key=lambda x: x[1], reverse=True)
        profiles = [p for p, _ in sized]
        order_desc = ", ".join(f"{p.split('_')[1] if '_' in p else p}:{n}" for p, n in sized[:8])
    else:
        order_desc = "(as given)"

    os.makedirs(args.output, exist_ok=True)
    log_dir = os.path.join(args.output, f"{args.cruise}_orchestrator_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(log_dir, exist_ok=True)

    total_cores = len(args.gpu_list) * args.n_cores_per_gpu
    log(f"Cruise={args.cruise}  GPUs={args.gpu_list}  cores/gpu={args.n_cores_per_gpu} (total {total_cores})")
    log(f"Profiles: {len(profiles)}   order(size): {order_desc}")
    log(f"Per-profile logs: {log_dir}")
    if passthrough:
        log(f"Forwarding to pipeline: {' '.join(passthrough)}")

    orch = Orchestrator(args, passthrough)
    t_start = time.time()
    try:
        orch.run(profiles, log_dir)
    except KeyboardInterrupt:
        log("Aborted.")
        return 130

    # Summary
    wall = time.time() - t_start
    ok = [r for r in orch.results if r["rc"] == 0]
    bad = [r for r in orch.results if r["rc"] != 0]
    log("=" * 60)
    log(f"DONE  wall={wall/60:.1f} min   ok={len(ok)}  failed={len(bad)}")
    for gpu in args.gpu_list:
        gres = [r for r in orch.results if r["gpu"] == gpu]
        busy = sum(r["seconds"] for r in gres)
        log(f"  GPU{gpu}: {len(gres)} profiles, busy {busy/60:.1f} min "
            f"({100*busy/wall:.0f}% of wall)")
    if bad:
        log("FAILED profiles:")
        for r in bad:
            log(f"  - {r['profile']} (GPU{r['gpu']}, rc={r['rc']})")
    log("=" * 60)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
