"""
AgentShield - Latency Benchmark
================================
runs 20 fake calls to see how slow the pipeline is:
  1. STT  — whisper takes its time
  2. EMB  — embedding the query
  3. RET  — grabbing stuff from pgvector
  4. LLM  — waiting for groq
  5. TOT  — total time it takes

saves output to data/benchmark_results.md

just run:
    python -m scripts.benchmark_latency
"""

import sys
import time
import statistics
import numpy as np
import soundfile as sf
import csv
import os
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")




TEST_QUERIES = [
    "I want a full refund for my broken router.",
    "My internet is not working since yesterday night.",
    "The internet light on my router is blinking red.",
    "I was charged twice for the same bill this month.",
    "How many days do I have to return the product?",
    "My WiFi is connected but there is no internet access.",
    "I want to speak to a supervisor right now.",
    "Can I pay my bill in EMI installments?",
    "The customer is threatening to go to consumer forum.",
    "How do I factory reset my router?",
    "My refund has not been processed after 10 days.",
    "I need to dispute an incorrect charge on my bill.",
    "What is the late payment fee if I miss the due date?",
    "Mera internet kaafi slow chal raha hai.",
    "Mujhe refund chahiye, product kaam nahi kar raha.",
    "Router ki light red blink kar rahi hai, kya karna chahiye?",
    "Bill mein galat charge aa gaya hai.",
    "Main case karunga agar yeh theek nahi hua.",
    "I need to cancel my subscription and get my money back.",
    "The speed test shows only 5 Mbps but I pay for 100 Mbps.",
]




def stats(values):
    if not values:
        return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    p95_idx = int(0.95 * n)
    return {
        "min":  round(min(sorted_v) * 1000),
        "max":  round(max(sorted_v) * 1000),
        "avg":  round(statistics.mean(sorted_v) * 1000),
        "p50":  round(sorted_v[n // 2] * 1000),
        "p95":  round(sorted_v[min(p95_idx, n - 1)] * 1000),
    }


def bar(ms, max_ms=3000, width=20):
    """draws a text progress bar, pretty cool"""
    pct = min(ms / max_ms, 1.0)
    filled = int(pct * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def rating(ms):
    """gives it a rating so we know if it's too slow"""
    if ms < 500:   return "EXCELLENT"
    if ms < 1000:  return "GOOD"
    if ms < 2000:  return "ACCEPTABLE"
    return "NEEDS IMPROVEMENT"





def load_components():
    print("\n  Loading components (one-time startup)...")

    import logging
    logging.getLogger("agentshield").setLevel(logging.CRITICAL)

    t0 = time.perf_counter()
    from src.audio.stt import Transcriber
    stt = Transcriber()
    stt_load = time.perf_counter() - t0
    print(f"    Whisper loaded        in {stt_load*1000:.0f} ms")

    t0 = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    from config.settings import Settings
    s = Settings()
    emb_model = SentenceTransformer(s.EMBEDDING_MODEL)
    emb_load = time.perf_counter() - t0
    print(f"    Embedding model loaded in {emb_load*1000:.0f} ms")

    t0 = time.perf_counter()
    from src.retrieval.hybrid import retrieve_context
    _ = retrieve_context("warmup", top_k=1)
    ret_load = time.perf_counter() - t0
    print(f"    pgvector warmed up    in {ret_load*1000:.0f} ms")

    t0 = time.perf_counter()
    from src.retrieval.generation import generate_answer
    gen_load = time.perf_counter() - t0
    print(f"    Groq LLM initialized  in {gen_load*1000:.0f} ms")

    print("  All components ready.\n")
    return stt, emb_model, retrieve_context, generate_answer





def make_dummy_audio(duration_s: float = 3.0):
    """just makes a beep for the stt to chew on"""
    sr = 16000
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio





def run_benchmark():
    print("=" * 62)
    print("  AgentShield - Latency Benchmark (20 Calls)")
    print("=" * 62)

    stt, emb_model, retrieve_context, generate_answer = load_components()

    dummy_audio = make_dummy_audio(3.0)

    stt_times, emb_times, ret_times, llm_times, tot_times = [], [], [], [], []
    rows = []

    print(f"  {'#':>2}  {'Query (truncated)':<38}  {'STT':>5}  {'EMB':>5}  {'RET':>5}  {'LLM':>5}  {'TOT':>5}")
    print(f"  {'-'*2}  {'-'*38}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")

    for i, query in enumerate(TEST_QUERIES):
        call_start = time.perf_counter()

        
        t0 = time.perf_counter()
        _ = stt.transcribe(dummy_audio)
        t_stt = time.perf_counter() - t0

        
        t0 = time.perf_counter()
        _ = emb_model.encode(query)
        t_emb = time.perf_counter() - t0

        
        t0 = time.perf_counter()
        context = retrieve_context(query, top_k=3)
        t_ret = time.perf_counter() - t0

        
        
        t0 = time.perf_counter()
        result = generate_answer(
            transcript=f"Customer: {query}",
            context=context if context else None
        )
        t_llm = time.perf_counter() - t0
        llm_failed = result is None

        t_tot = time.perf_counter() - call_start

        
        stt_times.append(t_stt)
        emb_times.append(t_emb)
        ret_times.append(t_ret)
        llm_times.append(t_llm)
        tot_times.append(t_tot)

        label = query[:38]
        llm_display = f"{t_llm*1000:>4.0f}ms" if not llm_failed else f"{'FAIL':>6}"
        print(
            f"  {i+1:>2}. {label:<38}  "
            f"{t_stt*1000:>4.0f}ms  "
            f"{t_emb*1000:>4.0f}ms  "
            f"{t_ret*1000:>4.0f}ms  "
            f"{llm_display}  "
            f"{t_tot*1000:>4.0f}ms"
        )

        rows.append({
            "call": i + 1,
            "query": query,
            "stt_ms": round(t_stt * 1000),
            "emb_ms": round(t_emb * 1000),
            "ret_ms": round(t_ret * 1000),
            "llm_ms": round(t_llm * 1000) if not llm_failed else "FAIL",
            "tot_ms": round(t_tot * 1000),
            "llm_failed": llm_failed,
        })

    
    
    
    st = {
        "STT":        stats(stt_times),
        "Embedding":  stats(emb_times),
        "Retrieval":  stats(ret_times),
        "LLM (Groq)": stats(llm_times),
        "TOTAL":      stats(tot_times),
    }

    print("\n" + "=" * 62)
    print("  PERFORMANCE SUMMARY (all values in ms)")
    print("=" * 62)
    print(f"  {'Stage':<14}  {'MIN':>5}  {'AVG':>5}  {'P50':>5}  {'P95':>5}  {'MAX':>5}  {'Rating':<18}")
    print(f"  {'-'*14}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*18}")
    for stage, s in st.items():
        r = rating(s["avg"])
        print(f"  {stage:<14}  {s['min']:>5}  {s['avg']:>5}  {s['p50']:>5}  {s['p95']:>5}  {s['max']:>5}  {r:<18}")

    print("=" * 62)
    avg_tot = st["TOTAL"]["avg"]
    print(f"\n  Average end-to-end latency: {avg_tot} ms")
    if avg_tot < 3000:
        print("  Result: REAL-TIME CAPABLE (< 3 seconds)")
    else:
        print("  Result: NEEDS OPTIMIZATION (> 3 seconds)")
    print()

    
    
    
    os.makedirs("data", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    
    md_path = "data/benchmark_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# AgentShield - Latency Benchmark Report\n\n")
        f.write(f"**Run Date:** {ts}  \n")
        f.write(f"**Total Test Calls:** 20  \n")
        f.write(f"**Whisper Model:** base (CPU, int8)  \n")
        f.write(f"**LLM:** Groq llama-3.3-70b-versatile  \n\n")

        f.write("## Performance Summary\n\n")
        f.write("| Stage | MIN (ms) | AVG (ms) | P50 (ms) | P95 (ms) | MAX (ms) | Rating |\n")
        f.write("|-------|----------|----------|----------|----------|----------|--------|\n")
        for stage, s in st.items():
            r = rating(s["avg"])
            f.write(f"| {stage} | {s['min']} | {s['avg']} | {s['p50']} | {s['p95']} | {s['max']} | {r} |\n")

        f.write(f"\n**Average end-to-end latency: {avg_tot} ms**\n\n")

        f.write("## Per-Call Results\n\n")
        f.write("| # | Query | STT (ms) | EMB (ms) | RET (ms) | LLM (ms) | TOTAL (ms) |\n")
        f.write("|---|-------|----------|----------|----------|----------|------------|\n")
        for r in rows:
            f.write(f"| {r['call']} | {r['query'][:50]} | {r['stt_ms']} | {r['emb_ms']} | {r['ret_ms']} | {r['llm_ms']} | {r['tot_ms']} |\n")

        f.write("\n## Key Observations\n\n")
        f.write(f"- **Fastest stage:** Embedding ({st['Embedding']['avg']} ms avg)\n")
        f.write(f"- **Slowest stage:** LLM / Groq ({st['LLM (Groq)']['avg']} ms avg)\n")
        f.write(f"- **STT (Whisper base CPU):** {st['STT']['avg']} ms avg\n")
        f.write(f"- **pgvector retrieval:** {st['Retrieval']['avg']} ms avg\n")
        f.write(f"- **Total pipeline:** {st['TOTAL']['avg']} ms avg\n")

    
    csv_path = "data/benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["call", "query", "stt_ms", "emb_ms", "ret_ms", "llm_ms", "tot_ms"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Results saved to:")
    print(f"    {md_path}")
    print(f"    {csv_path}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    run_benchmark()
