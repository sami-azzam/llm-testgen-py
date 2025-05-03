#!/usr/bin/env python3
"""
test_generation_pipeline.py
────────────────────────────────────────────────────────────────────────────
Generate and compare test-suites for Defects4J projects using

  • EvoSuite          (search-based)
  • GPT-4.1-nano      (LLM-based, JUnit 4)

Features
▪ resumable runs via  --resume-last
▪ token-saving GPT resume (skips already-generated tests)
▪ progress-bar for GPT requests (tqdm.asyncio)
▪ defensive error handling (never crashes on single-class failures)
"""

# ────────────────────────────── std-lib ────────────────────────────────────
import os, subprocess, random, shutil, asyncio, re, textwrap, xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import argparse, sys
from typing import List, Tuple

# ──────────────────────────── 3rd-party ────────────────────────────────────
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tabulate import tabulate
from tqdm import tqdm  # progress bar for async tasks

# ─────────────────────────── local helper ──────────────────────────────────
from java_finder import find_java11

# ══════════════════════════ CONFIGURATION ═════════════════════════════════
SEED: int                  = 2025
NUM_PROJECTS: int          = 1
BUG_REV: str               = "1f"
GPT_MODEL: str             = "o4-mini"
MAX_PARALLEL: int          = 200           # GPT calls in flight
EVOSUITE_MEM_MB: int       = 4096

ROOT: Path      = Path(__file__).resolve().parents[1]
RESULTS_DIR: Path = ROOT / "results"
EVOSUITE_JAR: Path = ROOT / "src/tools/evosuite.jar"

# ═══════════════════════ ENVIRONMENT SETUP ════════════════════════════════
load_dotenv(ROOT / ".env")                         # reads OPENAI_API_KEY, D4J_BIN, …
D4J_BIN = Path(os.getenv("D4J_BIN", ROOT / "data/defects4j/framework/bin")).expanduser().resolve()
os.environ["PATH"] = f"{D4J_BIN}:{os.environ['PATH']}"

ENV = {
    **os.environ,
    "JAVA_HOME": find_java11(),
    "D4J_COVERAGE": "jacoco",
    "TZ": "America/Los_Angeles"
}

client = AsyncOpenAI()                             # needs OPENAI_API_KEY env var

# ═══════════════════════─ UTILITY FUNCTIONS ─══════════════════════════════
def sh(cmd: List[str], cwd: Path | None = None, silent: bool = False,
       capture: bool = False) -> Tuple[int, str, str] | None:
    """Wrapper around subprocess.run()."""
    if not silent:
        print(" ".join(map(str, cmd)))
    if capture:
        p = subprocess.run(cmd, cwd=cwd, env=ENV, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode, p.stdout, p.stderr
    subprocess.run(cmd, cwd=cwd, env=ENV,
                   stdout=subprocess.DEVNULL if silent else None,
                   stderr=subprocess.STDOUT if silent else None,
                   check=True)
    return None

# ═══════════════════════─ PROJECT SAMPLING ─═══════════════════════════════
ALL_PROJECTS = [
    "Chart","Cli","Closure","Codec","Collections","Compress","Csv","Gson",
    "JacksonCore","JacksonDatabind","JacksonXml","Jsoup","JxPath","Lang",
    "Math","Mockito","Time"
]
random.Random(SEED).shuffle(ALL_PROJECTS)
PROJECTS = ALL_PROJECTS[:NUM_PROJECTS]

# ═══════════════════════─ ARGUMENT PARSING ─═══════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--resume-last", action="store_true", help="reuse newest runs_* folder")
ARGS = parser.parse_args()

existing_runs = sorted(RESULTS_DIR.glob("runs_*"))
RUN_DIR = existing_runs[-1] if (ARGS.resume_last and existing_runs) else \
          RESULTS_DIR / f"runs_{datetime.now():%Y%m%d_%H%M%S}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════─ DEFECTS4J HELPERS ─══════════════════════════════
def checkout_project(pid: str) -> Path:
    """Return (and create if missing) a pristine checkout directory."""
    base = RUN_DIR / f"{pid}_base"
    if not base.exists():
        sh(["defects4j","checkout","-p",pid,"-v",BUG_REV,"-w",base], silent=True)
    return base

def duplicate(src: Path, suffix: str) -> Path:
    """Copy directory to a new sibling with *suffix* appended."""
    dst = Path(f"{src}_{suffix}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst

# ═══════════════════════─ EVOSUITE ─═══════════════════════════════════════
def generate_evosuite_tests(work_dir: Path) -> bool:
    """Run EvoSuite; return True if successful."""
    if not EVOSUITE_JAR.is_file():
        print(f"   ⚠️  EvoSuite jar missing at {EVOSUITE_JAR}")
        return False

    cp = subprocess.check_output(
        ["defects4j","export","-p","cp.compile"], cwd=work_dir, env=ENV, text=True
    ).strip()
    test_src = work_dir / subprocess.check_output(
        ["defects4j","export","-p","dir.src.tests"], cwd=work_dir, env=ENV, text=True
    ).strip()
    test_src.mkdir(parents=True, exist_ok=True)

    rc, out, err = sh(
        ["java", f"-Xmx{EVOSUITE_MEM_MB}m", "-jar", EVOSUITE_JAR,
         "-projectCP", cp,
         "-generateSuite", "-class", ".*",
         "-Dtest_dir", test_src],
        cwd=work_dir, capture=True, silent=True
    )
    if rc:
        (work_dir / "evosuite_err.log").write_text(out + err)
        print("   ⚠️  EvoSuite failed (see evosuite_err.log)")
        return False
    return True


# ═══════════════════════─ GPT TEST GENERATION ─════════════════════════════

def generate_gpt_tests(work_dir: Path, skip_existing: bool) -> None:
    """Generate tests in parallel with progress bar; skip/ignore failures."""
    classes = subprocess.check_output(
        ["defects4j","export","-p","classes.relevant"],
        cwd=work_dir, env=ENV, text=True
    ).splitlines()

    test_src = work_dir / subprocess.check_output(
        ["defects4j","export","-p","dir.src.tests"],
        cwd=work_dir, env=ENV, text=True
    ).strip()
    test_src.mkdir(parents=True, exist_ok=True)

    tasks = []
    for fqcn in classes:
        dst_file = test_src / f"{fqcn.split('.')[-1]}GptTest.java"
        if skip_existing and dst_file.exists():
            continue
        tasks.append((fqcn, dst_file))

    if not tasks:
        print("   ⏩  all GPT tests present")
        return


    # 1) Get the relative source‐root (e.g. "gson/src/main/java")
    rel_src = subprocess.check_output(
        ["defects4j", "export", "-p", "dir.src.classes"],
        cwd=work_dir, env=ENV, text=True
    ).strip()
    src_root = work_dir / rel_src

    # 2) Get all relevant class‐names (FQCNs)
    fqcn_list = subprocess.check_output(
        ["defects4j", "export", "-p", "classes.relevant"],
        cwd=work_dir, env=ENV, text=True
    ).splitlines()

    # 3) Build a mapping FQCN → actual .java file on disk
    fqcn_to_file: dict[str,Path] = {}
    for fqcn in fqcn_list:
        path = src_root / Path(fqcn.replace(".", "/") + ".java")
        if path.is_file():
            fqcn_to_file[fqcn] = path
        else:
            print(f"   • ⚠️  source file for {fqcn} not found at {path}")

    async def _ask_gpt_for_test(fqcn: str, sem: asyncio.Semaphore) -> tuple[str, str] | None:
        """Return (fqcn, java_source) or None if unusable."""
        file_path = fqcn_to_file.get(fqcn)
        if not file_path or not file_path.is_file():
            print(f"     • ⚠️  source for {fqcn} not found!")
            return None
        source = file_path.read_text()
        
        prompt = f"""
    Here is the *exact* source code of `{fqcn}`:

    ```java
    {source}
```

    **Task**
    1. Silently analyse the above class's public API, method behaviors, edge cases, and branches.
    2. Then output exactly one JUnit 4 test-class that maximizes branch coverage for this code

    **Rules**
    - the code **MUST BE compatible with Java 6** (no diamond operators `<>`, no try-with-resources, etc.).  
    - Use only org.junit.Assert.*.
    - No imports beyond JUnit 4 (`import org.junit.Test;`, `import static org.junit.Assert.*;`).  
    - No comments or explanations.
    - Provide *only* the final test code inside a java code-block.
    """
        async with sem:
            rsp = await client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "system", "content": "You are a senior QA engineer AI Assistant. You only reply with code."},{"role": "user", "content": prompt}],
            )

        # Get the raw assistant reply
        msg = rsp.choices[0].message.content
        tokens = rsp.usage.total_tokens
        # 1) Remove any leading ```java (or ```)
        msg = re.sub(r'^```(?:java)?\s*', '', msg)
        # 2) Remove any trailing ```
        msg = re.sub(r'\s*```$', '', msg)
        # 3) Dedent and trim
        code = textwrap.dedent(msg).strip()

        if "class" not in code or "@Test" not in code:
            print(f"     • ⚠️  no usable code for {fqcn} (skipped)")
            print("--- Code was", code)
            print("--- Tokens Used", tokens)
            return None
        return fqcn, code, tokens

    async def _run():
        # ── grab compile class-path once ────────────────────────────────
        compile_cp = subprocess.check_output(
            ["defects4j", "export", "-p", "cp.compile"],
            cwd=work_dir, env=ENV, text=True
        ).strip()

        sem   = asyncio.Semaphore(MAX_PARALLEL)
        coros = [_ask_gpt_for_test(fqcn, sem) for fqcn, _ in tasks]

        pbar        = tqdm(total=len(coros), desc="   GPT", unit="test")
        generated   = 0
        compile_ok  = 0
        compile_bad = 0
        tokens_used = 0
        for fut in asyncio.as_completed(coros):
            res = await fut
            if res is not None:
                fqcn, code, tok = res
                tokens_used += tok
                dst = test_src / f"{fqcn.split('.')[-1]}GptTest.java"
                dst.write_text(code)
                generated += 1

            pbar.update(1)

        pbar.close()
        print(f"   ✓ generated {generated}/{len(tasks)} GPT tests  "
              f"({compile_ok} compile-ok, {compile_bad} renamed)"
              f" \n  Total tokens used: {tokens_used}")

    asyncio.run(_run())


# ───────────────────────── helper: prune bad GPT tests ────────────────────
def prune_noncompiling(work_dir: Path) -> None:
    """
    For every generated *GptTest.java, try to compile it in isolation with
    the project's compile class-path.  If compilation fails, rename the file
    to *.java.disabled so Ant/Defects4J will ignore it.
    """
    # class-path once
    cp = subprocess.check_output(
        ["defects4j", "export", "-p", "cp.compile"],
        cwd=work_dir, env=ENV, text=True
    ).strip()

    # test-src dir once
    test_src = work_dir / subprocess.check_output(
        ["defects4j", "export", "-p", "dir.src.tests"],
        cwd=work_dir, env=ENV, text=True
    ).strip()

    for java in test_src.rglob("*GptTest.java"):
        rc = subprocess.run(
            ["javac", "-classpath", cp, str(java)],
            cwd=work_dir, env=ENV,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        if rc != 0:
            bad = java.with_suffix(".java.disabled")
            java.rename(bad)
            print(f"   • ⚠️  {java.name} does NOT compile – renamed to {bad.name}")

# ═══════════════════════─ COVERAGE ─═══════════════════════════════════════
def coverage_stats(work_dir: Path, tag: str, proj: str) -> dict:
    """
    Compile, run tests, collect JaCoCo coverage.  Always returns a result dict:
      {project, generator, line_%, branch_%, compiled, tests_passed}
    The pipeline will not crash even if compilation or tests fail.
    """
    # ── NEW: strip out non-compiling GPT files first ─────────────────────
    if tag.startswith("GPT"):
        prune_noncompiling(work_dir)
    # --- compile -----------------------------------------------------------
    compile_rc = subprocess.run(
        ["defects4j", "compile"], cwd=work_dir, env=ENV
    ).returncode
    if compile_rc != 0:
        print(f"   ⚠️  {tag} tests do NOT compile")
        return {"project": proj, "generator": tag,
                "line_%": 0.0, "branch_%": 0.0,
                "compiled": False, "tests_passed": False}

    # --- run developer + generated tests -----------------------------------
    test_rc = subprocess.run(
        ["defects4j", "test"], cwd=work_dir, env=ENV
    ).returncode
    if test_rc != 0:
        print(f"   ⚠️  some {tag} tests FAILED (non-zero exit)")

    # try coverage regardless; it often works even with failing tests
    subprocess.run(
        ["defects4j", "coverage"], cwd=work_dir, env=ENV, stdout=subprocess.DEVNULL
    )

    # --- parse JaCoCo, fallback to zeros -----------------------------------
    cov_xml = work_dir / "coverage.xml"
    if not cov_xml.is_file():
        return {"project": proj, "generator": tag,
                "line_%": 0.0, "branch_%": 0.0,
                "compiled": True, "tests_passed": (test_rc == 0)}

    root = ET.parse(cov_xml).getroot()
    def pct(kind: str) -> float:
        n = root.find(f"./counter[@type='{kind}']")
        if n is None:
            return 0.0
        cov = int(n.get("covered")); miss = int(n.get("missed"))
        return round(100 * cov / (cov + miss), 2) if cov + miss else 0.0

    return {"project": proj, "generator": tag,
            "line_%": pct("LINE"), "branch_%": pct("BRANCH"),
            "compiled": True, "tests_passed": (test_rc == 0)}

# ═══════════════════════─ MAIN LOOP ─══════════════════════════════════════
results: List[dict] = []

for pid in PROJECTS:
    print(f"\n════ {pid} ════")
    base = checkout_project(pid)

    # ——— EvoSuite
    evo_dir = Path(f"{base}_evo")
    if not (ARGS.resume_last and evo_dir.exists()):
        evo_dir = duplicate(base, "evo")
        print("  🧬 EvoSuite generating…")
        generate_evosuite_tests(evo_dir)
    else:
        print("  🧬 EvoSuite reuse")
    results.append(coverage_stats(evo_dir, "EvoSuite", pid))

    # ——— GPT
    gpt_dir = Path(f"{base}_gpt")
    if ARGS.resume_last and gpt_dir.exists():
        print("  🤖 GPT resume…")
        generate_gpt_tests(gpt_dir, skip_existing=True)
    else:
        gpt_dir = duplicate(base, "gpt")
        print(f"  🤖 GPT generating ({MAX_PARALLEL} parallel)…")
        generate_gpt_tests(gpt_dir, skip_existing=False)
    results.append(coverage_stats(gpt_dir, GPT_MODEL, pid))

# ═══════════════════════─ SUMMARY ─════════════════════════════════════════
df = pd.DataFrame(results)
df.to_csv(RUN_DIR / "coverage.csv", index=False)

print("\n📊  Coverage comparison")
print(tabulate(
    df.pivot(index="project", columns="generator",
             values=["line_%", "branch_%"]).round(2),
    headers="keys", tablefmt="github"
))
