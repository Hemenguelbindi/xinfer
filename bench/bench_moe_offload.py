#!/usr/bin/env python3
"""Benchmark MoE expert offload (XINFER_MOE_OFFLOAD) across GPU expert fractions.

Runs xinfer once per fraction (0 = no offload, the baseline) and writes a Markdown
table plus JSON with the environment, so results from Windows, WSL and Docker can
be compared side by side.

  python3 bench/bench_moe_offload.py --bin target/release/xinfer \
      --model /models/Qwen3.6-35B-A3B-NVFP4 --rates 0,0.1,0.2,0.35 --out bench/qwen36

--mode server (default) starts the OpenAI-compatible server, sends one warm-up and
one streamed request, and timestamps every streamed chunk: time to first token,
decode tok/s and inter-token latency percentiles (stalls show up in p99/max).
--mode cli uses `--prompts` and xinfer's own throughput numbers.

Greedy decoding is used, so every successful run must produce the same text.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PATTERNS = {
    "free_vram_gb": r"free ([0-9.]+) GB, kv_fraction",
    "kv_tokens": r"max KV-cache tokens (\d+)",
    "prefill_tps": r"(?:Prompt tokens: \d+ in [0-9.]+s|total tokens in [0-9.]+s) \(([0-9.]+) tokens/s\)",
    "prompt_tokens": r"(\d+) total tokens in",
    "decode_tps": r"Decoded tokens: \d+ in [0-9.]+s \(([0-9.]+) tokens/s\)",
    "hit_rate_pct": r"expert hit rate ([0-9.]+)%",
}


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment(binary):
    gpu = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max",
              "--format=csv,noheader"])
    in_docker = os.path.exists("/.dockerenv")
    in_wsl = "microsoft" in platform.release().lower()
    return {
        "platform": "docker" if in_docker else ("wsl" if in_wsl else platform.system().lower()),
        "os": platform.platform(),
        "cpu": platform.processor() or sh(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]),
        "gpu": gpu,
        "binary": binary,
        "git": sh(["git", "rev-parse", "--short", "HEAD"]),
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }


def parse(log):
    res = {"cuda_graph": "Cuda graph captured" in log}
    for key, pat in PATTERNS.items():
        found = re.findall(pat, log)
        if found:
            res[key] = float(found[-1])
    return res


def failure(log):
    errors = [l for l in log.splitlines() if re.search(r"ERROR|Error:|panicked|out of memory|failed", l)]
    last = errors[-1] if errors else (log.strip().splitlines() or [""])[-1]
    return last.split("Error: ")[-1][-200:]


def offload_env(rate):
    env = dict(os.environ)
    env.pop("XINFER_MOE_OFFLOAD", None)
    if rate != 0:
        env["XINFER_MOE_OFFLOAD"] = str(rate)
    return env


def base_cmd(args):
    return [args.bin, "--m", args.model, "--max-model-len", str(args.max_model_len),
            "--disable-reasoning", *args.xinfer_args.split()]


def run_cli(args, rate):
    cmd = base_cmd(args) + ["--prompts", args.prompt, "--max-tokens", str(args.max_tokens),
                            "--temperature", "0", "--top-k", "1"]
    try:
        p = subprocess.run(cmd, env=offload_env(rate), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=args.timeout)
        log = ANSI.sub("", p.stdout + p.stderr)
    except subprocess.TimeoutExpired as e:
        log = ANSI.sub("", (e.stdout or b"").decode(errors="replace"))
    res = parse(log)
    m = re.search(r"Response: (.*?)\n\S*\s*INFO xinfer: --- Performance Metrics", log, re.S)
    if m:
        res["response"] = m.group(1).strip()
    return res, log


def stream_chat(base, prompt, max_tokens, timeout):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "top_k": 1, "stream": True, "enable_thinking": False}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    sent, stamps, text = time.perf_counter(), [], []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            delta = (json.loads(data).get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                stamps.append(time.perf_counter())
                text.append(delta)
    return sent, stamps, "".join(text)


WORDS = ["amber", "basalt", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper"]


def long_prompt(tokens):
    """~`tokens` tokens of numbered facts plus a question about the middle one (needle check)."""
    n = max(1, tokens // 23)  # measured: ~23 Qwen tokens per fact line
    facts = "\n".join(f"Fact {i}: the code word for item {i} is {WORDS[i % 10]}-{i * 7 % 1000}." for i in range(n))
    i = n // 2
    question = f"What is the code word for item {i}? Answer with the code word only."
    return f"{facts}\n\n{question}", f"{WORDS[i % 10]}-{i * 7 % 1000}"


def percentile(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def run_server(args, rate):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(base_cmd(args) + ["--server", f"127.0.0.1:{port}"], env=offload_env(rate),
                            stdout=log_file, stderr=subprocess.STDOUT)
    res = {}
    try:
        deadline = time.time() + args.timeout
        while time.time() < deadline and proc.poll() is None:
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=2).close()
                break
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        if proc.poll() is None:
            stamps = []
            try:
                stream_chat(base, "Say OK.", 8, args.timeout)  # warm-up; a different prompt so prefix cache can't help
                sent, stamps, text = stream_chat(base, args.prompt, args.max_tokens, args.timeout)
            except urllib.error.HTTPError as e:
                res["http_error"] = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
            if stamps:
                gaps = [1e3 * (b - a) for a, b in zip(stamps, stamps[1:])]
                res.update(response=text, ttft_ms=1e3 * (stamps[0] - sent), chunks=len(stamps))
                if gaps:
                    res.update(decode_tps=len(gaps) / ((stamps[-1] - stamps[0]) or 1e-9),
                               itl_p50_ms=percentile(gaps, 0.5), itl_p95_ms=percentile(gaps, 0.95),
                               itl_p99_ms=percentile(gaps, 0.99), itl_max_ms=max(gaps))
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_file.seek(0)
        log = ANSI.sub("", log_file.read())
        log_file.close()
    parsed = parse(log)
    parsed.pop("decode_tps", None)  # the streamed measurement is used in server mode
    parsed.update(res)
    return parsed, log


def run(args, rate):
    start = time.time()
    res, log = (run_server if args.mode == "server" else run_cli)(args, rate)
    res.update(rate=rate, wall_s=round(time.time() - start, 1))
    if args.needle:
        res["needle_found"] = args.needle in res.get("response", "")
    if res.get("response"):
        res["response_sha"] = hashlib.sha256(res["response"].encode()).hexdigest()[:12]
    else:
        res["error"] = res.pop("http_error", None) or failure(log)
    if args.keep_logs:
        with open(f"{args.out}.rate{rate}.log", "w", encoding="utf-8") as f:
            f.write(log)
    return res


def markdown(env, results, args):
    ok = [r for r in results if "response_sha" in r]
    same = len({r["response_sha"] for r in ok}) <= 1
    name = os.path.basename(args.model.rstrip("/\\"))
    lines = [
        f"### {name} on {env['platform']}",
        "",
        f"- GPU: {env['gpu']}",
        f"- OS: {env['os']}; CPU: {env['cpu']}",
        f"- xinfer {env['git']}, {env['date']}; {args.mode} mode, up to {args.max_tokens} decoded tokens,"
        f" greedy, max-model-len {args.max_model_len} {args.xinfer_args}".rstrip(),
        "",
        "| GPU experts | Status | Free VRAM after load | KV tokens | Prompt tokens | Prefill tok/s | TTFT ms"
        " | Decode tok/s | Inter-token p50 / p99 / max ms | CUDA graph |" + (" Needle |" if args.needle else ""),
        "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if args.needle else ""),
    ]
    for r in results:
        label = {0: "all (no offload)", "auto": "auto"}.get(r["rate"]) or f"{r['rate']:.0%}"
        status = "ok" if "response_sha" in r else f"failed: {r.get('error', '?')}"

        def fmt(k, f="{:.1f}"):
            return f.format(r[k]) if k in r else "-"

        itl = (f"{r['itl_p50_ms']:.1f} / {r['itl_p99_ms']:.1f} / {r['itl_max_ms']:.1f}"
               if "itl_p50_ms" in r else "-")
        needle = (" yes |" if r.get("needle_found") else " no |") if args.needle else ""
        lines.append(f"| {label} | {status} | {fmt('free_vram_gb', '{:.2f} GB')} | {fmt('kv_tokens', '{:.0f}')} | "
                     f"{fmt('prompt_tokens', '{:.0f}')} | {fmt('prefill_tps')} | {fmt('ttft_ms', '{:.0f}')} | "
                     f"{fmt('decode_tps')} | {itl} | {'yes' if r['cuda_graph'] else 'no'} |{needle}")
    lines += ["", f"Greedy outputs identical across successful runs: **{'yes' if same else 'NO'}**"
              f" ({len(ok)} runs)", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rates", default="0,0.1,0.2,0.35", help="0 = baseline without offload; auto = fill free VRAM")
    ap.add_argument("--mode", choices=["server", "cli"], default="server")
    ap.add_argument("--prompt", default="Explain in three sentences what a Mixture-of-Experts model is.")
    ap.add_argument("--prompt-tokens", type=int, default=0, help="generate a long needle prompt of ~N tokens")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--xinfer-args", default="", help='extra flags, e.g. "--kvcache-dtype fp8"')
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="bench/moe_offload")
    ap.add_argument("--keep-logs", action="store_true")
    args = ap.parse_args()
    args.needle = ""
    if args.prompt_tokens:
        args.prompt, args.needle = long_prompt(args.prompt_tokens)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    env = environment(args.bin)
    results = []
    for rate in [x if x == "auto" else float(x) for x in args.rates.split(",")]:
        print(f"[bench] XINFER_MOE_OFFLOAD={rate or 'off'} ...", file=sys.stderr, flush=True)
        r = run(args, rate)
        print(f"[bench]   -> decode {r.get('decode_tps', '-')} tok/s, {r.get('error', 'ok')}",
              file=sys.stderr, flush=True)
        results.append(r)

    md = markdown(env, results, args)
    with open(f"{args.out}.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump({"env": env, "args": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
    print(md)


if __name__ == "__main__":
    main()
