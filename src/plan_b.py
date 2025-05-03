#!/usr/bin/env python3
"""
test_generation_pipeline_planB.py
────────────────────────────────────────────────────────────────────────────
Benchmark Defects4J bugs with

  • EvoSuite          (search-based, Java 6 target)
  • GPT               (LLM-based, user-supplied prompt)

Plan B keeps every build at Java 6 and avoids Cobertura by compiling with
`defects4j compile`.  JaCoCo is used for coverage.
"""

# ─────────────── std-lib ──────────────────────────────────────────────────
from __future__ import annotations
import os, subprocess, random, shutil, asyncio, re, textwrap, xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Tuple

# ─────────────── third-party ──────────────────────────────────────────────
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tabulate import tabulate
from tqdm import tqdm
import csv


# ─────────────── helper ───────────────────────────────────────────────────
from java_finder import find_java11          # user-supplied util

# ═════════════════ config ════════════════════════════════════════════════
SEED, NUM_PROJECTS, BUG_REV = 2025, 5, "1f"
GPT_MODEL         = "o4-mini"            # put any model name; prompt left blank
MAX_PARALLEL_GPT  = 100
EVOSUITE_MEM_MB   = 4096

ROOT         = Path(__file__).resolve().parents[1]
RESULTS_DIR   = ROOT / "results"
EVOSUITE_JAR  = ROOT / "src/tools/evosuite.jar"
total_tokens = 0

# ═══════════════ environment ═════════════════════════════════════════════
load_dotenv(ROOT / ".env")
D4J_BIN = Path(os.getenv("D4J_BIN", ROOT / "data/defects4j/framework/bin")).expanduser().resolve()
os.environ["PATH"] = f"{D4J_BIN}:{os.environ['PATH']}"

ENV = {**os.environ,
       "JAVA_HOME": find_java11(),      # JDK 11+ fine; we emit -source 6 code
       "D4J_COVERAGE": "jacoco",
       "TZ": "America/Los_Angeles"}

client = AsyncOpenAI()                  # requires OPENAI_API_KEY

# ═══════════════ helpers ═════════════════════════════════════════════════
def sh(cmd: List[str], cwd: Path | None = None, quiet=False,
       capture=False) -> Tuple[int,str,str] | None:
    if not quiet:
        print(" ".join(map(str,cmd)))
    if capture:
        p = subprocess.run(cmd, cwd=cwd, env=ENV, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode, p.stdout, p.stderr
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True,
                   stdout=subprocess.DEVNULL if quiet else None,
                   stderr=subprocess.STDOUT if quiet else None)

# sample Defects4J projects
ALL = ["Chart","Cli","Closure","Codec","Collections","Compress","Csv","Gson",
       "JacksonCore","JacksonDatabind","JacksonXml","Jsoup","JxPath","Lang",
       "Math","Mockito","Time"]
random.Random(SEED).shuffle(ALL)
PROJECTS = ALL[:NUM_PROJECTS]

# argparse
ap = argparse.ArgumentParser()
ap.add_argument("--resume-last", action="store_true")
ARGS = ap.parse_args()

runs = sorted(RESULTS_DIR.glob("runs_*"))
RUN_DIR = runs[-1] if (ARGS.resume_last and runs) else \
          RESULTS_DIR/f"runs_{datetime.now():%Y%m%d_%H%M%S}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════ Defects4J wrappers ═══════════════════════════════════════
def checkout(pid: str) -> Path:
    base = RUN_DIR/f"{pid}_base"
    if not base.exists():
        sh(["defects4j","checkout","-p",pid,"-v",BUG_REV,"-w",base], quiet=False)
    return base

def clone(src: Path, suf: str) -> Path:
    dst = Path(f"{src}_{suf}")
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src, dst); return dst

# ═══════════════ EvoSuite (Java 6) ═══════════════════════════════════════
def run_evosuite(work: Path) -> None:
    # 0) make sure sources & deps are compiled
    sh(["defects4j", "compile"], cwd=work)

    # 1) full class-path (classes + libs)
    cp = subprocess.check_output(
        ["defects4j", "export", "-p", "cp.test"],
        cwd=work, env=ENV, text=True).strip()

    test_src = work / subprocess.check_output(
        ["defects4j", "export", "-p", "dir.src.tests"],
        cwd=work, env=ENV, text=True).strip()
    test_src.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java", f"-Xmx{EVOSUITE_MEM_MB}m",
        "-jar", str(EVOSUITE_JAR),
        "-projectCP", cp,
        "-generateSuite", "-class", ".*",
        f"-Dtest_dir={str(test_src)}"
    ]
    rc, out, err = sh(cmd, cwd=work, capture=True, quiet=True)
    if rc:
        (work/"evosuite_err.log").write_text(out + err)
        print("   ⚠️  EvoSuite failed (see evosuite_err.log)")

# ═══════════════ GPT test generation ═════════════════════════════════════
PROMPT = f"""
Here is the *exact* source code of `{{FQCN}}`:

```java
{{CODE}}
```

**Task**
1. Silently analyse the above class's public API, method behaviors, edge cases, and branches.
2. Then output exactly one JUnit 4 test-class that maximizes branch coverage for this code

**Rules**
- the output **MUST BE a valid JUnit 4 test class**, and compile cleanly with Java 6.
- Code **must compile with `javac -source 6 -target 6`**.  
  Therefore **do NOT use any feature introduced in Java 7 or later**, including but not limited to:  
  • diamond operator `<>` or generic type inference  
  • try-with-resources (`try (...) {{}}`)  
  • multi-catch (`catch (A | B e)`) or re-throw type inference  
  • strings or enums in `switch`  
  • binary / underscore numeric literals (`0b1010`, `1_000`)  
  • lambda expressions, method references, `java.util.stream.*` API  
  • default / static interface methods  
  • `var`, multi-line text blocks, or records  
  • annotations on types (`@Nonnull String`) or `@Override` on interface methods 
  • AGAIN, **DO NOT use these or any feature introduced in Java 7 or later!**.
- Use only org.junit.Assert.*.
- No imports beyond JUnit 4 (`import org.junit.Test;`, `import static org.junit.Assert.*;`).  
- No comments or explanations.
- Provide *only* the final test code inside a java code-block.
    """

def sanitize(fqcn: str, code: str) -> str:
    newname = fqcn.split('.')[-1]+"GptTest"
    return re.sub(r'\bclass\s+\w+', f'class {newname}', code, count=1)

def gpt_tests(work: Path, skip_existing: bool):
    tests_rel = subprocess.check_output(
        ["defects4j","export","-p","dir.src.tests"], cwd=work, env=ENV, text=True).strip()
    tests_dir = work/tests_rel; tests_dir.mkdir(parents=True, exist_ok=True)

    classes = subprocess.check_output(
        ["defects4j","export","-p","classes.relevant"], cwd=work, env=ENV, text=True).splitlines()
    to_run = [(c, tests_dir/f"{c.split('.')[-1]}GptTest.java")
              for c in classes if not (skip_existing and (tests_dir/f"{c.split('.')[-1]}GptTest.java").exists())]
    if not to_run: print("   ⏩  all GPT tests present"); return None, None, 0

    src_root = work / subprocess.check_output(
        ["defects4j","export","-p","dir.src.classes"], cwd=work, env=ENV, text=True).strip()
    src_map = {c: src_root/Path(c.replace('.','/')+'.java') for c in classes}

    async def _ask(fqcn:str, sem:asyncio.Semaphore):
        path = src_map.get(fqcn); 
        if not path or not path.is_file(): return None, None, 0
        user_prompt = PROMPT.replace("{CODE}", path.read_text()).replace("{FQCN}", fqcn)
        tokens = 0
        async with sem:
            rsp = await client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role":"user","content":user_prompt}],
            )
            if rsp.usage:
                tokens = rsp.usage.total_tokens
        text = rsp.choices[0].message.content
        text = re.sub(r'^```(?:java)?\s*','',text).rstrip("` \n")
        text = sanitize(fqcn, textwrap.dedent(text))
        return fqcn, text, tokens

    async def _run():
        global total_tokens
        sem = asyncio.Semaphore(MAX_PARALLEL_GPT)
        coros = [_ask(fqcn, sem) for fqcn,_ in to_run]
        
        bar = tqdm(total=len(coros), desc="   GPT", unit="test")
        for fut in asyncio.as_completed(coros):
            res = await fut
            if res:
                fqcn, src, tokens = res
                (tests_dir/f"{fqcn.split('.')[-1]}GptTest.java").write_text(src)
                if tokens : total_tokens += tokens
            bar.update(1)
        bar.close()
    asyncio.run(_run())

# ═══════════════ prune non-compiling GPT tests (Java 6) ═══════════════════
# ───────────────────── helper: quarantine non-compiling GPT tests ─────────
def prune(work: Path) -> None:
    """
    Compile every *GptTest.java individually with -source/-target 6.
    • Prints  ✓ file   when it compiles.
    • Prints ⚠️ file disabled  + the first lines of javac output when it fails,
      then renames the file to *.java.disabled so Ant ignores it.
    Repeats until no active GPT test remains uncompilable.
    """
    # constant resources
    cp = subprocess.check_output(
        ["defects4j", "export", "-p", "cp.test"],
        cwd=work, env=ENV, text=True).strip()

    tests_dir = work / subprocess.check_output(
        ["defects4j", "export", "-p", "dir.src.tests"],
        cwd=work, env=ENV, text=True).strip()
    kept = 0
    removed = 0
    while True:
        active = sorted(tests_dir.rglob("*GptTest.java"))
        if not active:
            print("   ⚠️  no active GPT tests left after pruning")
            return

        progress_made = False

        for src in active:
            proc = subprocess.run(
                ["javac", "-source", "6", "-target", "6",
                 "-classpath", cp, str(src)],
                cwd=work, env=ENV,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
            )
      
            if proc.returncode == 0:
                kept += 1
                # print(f"   ✓ {src.relative_to(work)}")
                continue
            removed += 1
            # ── compile failed ──
            # print(f"   ⚠️  {src.relative_to(work)} disabled")
            # print at most 20 lines so logs stay readable
            # for line in proc.stdout.splitlines()[:20]:
            #     print(f"         {line}")
            bad = src.with_suffix(".java.disabled")
            src.rename(bad)
            progress_made = True
            # restart scan: removal might unblock others
            break
        
        if not progress_made:
            # nothing failed in this iteration → all remaining files compile
            break
    print(f"   ⚠️  {removed} GPT tests disabled, {kept} kept, {kept + removed} total")
        

# ═══════════════ coverage (compile) ═════════════════════════════
import csv

# def coverage(work: Path, tag: str, proj: str) -> dict:
#     """
#     Compile, run tests, run d4j-coverage on *all* classes, then read summary.csv
#     and return {'project','generator','line_%','branch_%'}.
#     """
#     # 1) prune GPT work-tree if requested
#     if tag.startswith("gpt"):
#         prune(work)
#     const dir = RESULTS_DIR
#     # 2) compile sources
#     if subprocess.run(["defects4j", "compile"],
#                       cwd=work, env=ENV).returncode:
#         return {"project": proj, "generator": tag,
#                 "line_%": 0.0, "branch_%": 0.0}

#     # 3) make a list of *all* production classes once per checkout
#     inst_file = work / "all_classes.txt"
#     if not inst_file.exists():
#         src_root = work / "src/main/java"
#         classes = [".".join(f.relative_to(src_root)
#                                .with_suffix('')
#                                .parts)
#                    for f in src_root.rglob("*.java")]
#         inst_file.write_text("\n".join(sorted(classes)))

#     # 4) run the test suite (silenced) – needed to populate Cobertura DB
#     subprocess.run(["defects4j", "test"],
#                    cwd=work, env=ENV,
#                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     # 5) run coverage for *all* classes
#     subprocess.run(["defects4j", "coverage",
#                     "-i", str(inst_file)],
#                    cwd=work, env=ENV,
#                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     # 6) parse summary.csv (still your preferred format)
#     csv_file = work / "summary.csv"
#     if not csv_file.is_file():
#         return {"project": proj, "generator": tag,
#                 "line_%": 0.0, "branch_%": 0.0}

#     with csv_file.open(newline="") as fh:
#         row = next(csv.DictReader(fh))
#         lt, lc = int(row["LinesTotal"]),      int(row["LinesCovered"])
#         ct, cc = int(row["ConditionsTotal"]), int(row["ConditionsCovered"])

#     return {"project": proj, "generator": tag,
#             "line_%":   round(100 * lc / lt, 2) if lt else 0.0,
#             "branch_%": round(100 * cc / ct, 2) if ct else 0.0}

# def coverage(work: Path, tag: str, proj: str) -> dict:
#     """Run compilation, tests, then parse Defects4J's summary.csv.

#     Returns a dict with   {project, generator, line_%, branch_%}
#     """
#     # --- 1. prune GPT worktree if requested
#     # --- 1. prune GPT worktree if requested
#     if tag.startswith("gpt"):
#         prune(work)                  # your existing helper

#     print(" WORK DIR::: ", work)
#     # --- 2. compile the project
#     if subprocess.run(["defects4j", "compile"], cwd=work, env=ENV).returncode:
#         return {"project": proj, "generator": tag, "line_%": 0.0, "branch_%": 0.0}

#     # --- 3. execute tests & produce coverage *summary* file
#     odd = subprocess.run(["defects4j", "test"],      cwd=work, env=ENV,
#                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
#     if odd.returncode != 0:
#         print("   ⚠️  test execution failed")
#         print(odd.stderr)
#         return {"project": proj, "generator": tag, "line_%": 0.0, "branch_%": 0.0}
    
#     subprocess.run(["defects4j", "coverage" "-w", "."],  
#                    cwd=work, env=ENV,
#                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     # --- 4. parse summary.csv -------------------------------------------------
#     csv_file = work / "summary.csv"
#     if not csv_file.exists():
#         # fall back on 0 % (or keep your old XML parser as a secondary path)
#         return {"project": proj, "generator": tag, "line_%": 0.0, "branch_%": 0.0}

#     with csv_file.open(newline="") as fh:
#         reader = csv.DictReader(fh)
#         row    = next(reader)  # only one data row
#         lines_total  = int(row["LinesTotal"])
#         lines_cov    = int(row["LinesCovered"])
#         cond_total   = int(row["ConditionsTotal"])
#         cond_cov     = int(row["ConditionsCovered"])

#     line_pct   = round(100 * lines_cov  / lines_total,  2) if lines_total  else 0.0
#     branch_pct = round(100 * cond_cov   / cond_total,  2) if cond_total   else 0.0

#     return {"project": proj,
#             "generator": tag,
#             "line_%": line_pct,
#             "branch_%": branch_pct}
def coverage(work: Path, tag: str, proj: str) -> dict:
    if tag.startswith("gpt"):
        prune(work)                      # your existing helper

    # 1 · compile
    if subprocess.run(["defects4j", "compile"],
                      cwd=work, env=ENV).returncode:
        return {"project": proj, "generator": tag,
                "line_%": 0.0, "branch_%": 0.0}

    # 2 · build — once per work-tree — a list of *all* classes to instrument
    instr = work / "instrument_classes.txt"
    if not instr.exists():
        classes_root = work / "build" / "classes"     # <-- Ant build dir
        if not classes_root.exists():                 # fall-back to Maven layout
            classes_root = work / "target" / "classes"

        with instr.open("w") as fh:
            for cls in classes_root.rglob("*.class"):
                # convert path to fully-qualified name  e.g. com/google/Foo.class → com.google.Foo
                fqcn = ".".join(cls.relative_to(classes_root)
                                    .with_suffix("").parts)
                fh.write(fqcn + "\n")

    # 3 · run tests + coverage on *all* those classes
    subprocess.run(["defects4j", "test"],
                   cwd=work, env=ENV,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run(["defects4j", "coverage", "-i", str(instr)],
                   cwd=work, env=ENV,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 4 · read Defects4J’s summary.csv (still generated next to coverage.xml)
    csv_file = work / "summary.csv"
    if not csv_file.exists():
        return {"project": proj, "generator": tag,
                "line_%": 0.0, "branch_%": 0.0}

    with csv_file.open(newline="") as fh:
        row = next(csv.DictReader(fh))
    line_pct   = round(100 * int(row["LinesCovered"])    / int(row["LinesTotal"]),     2)
    branch_pct = round(100 * int(row["ConditionsCovered"])/ int(row["ConditionsTotal"]), 2)

    return {"project": proj, "generator": tag,
            "line_%": line_pct, "branch_%": branch_pct}

# ═══════════════ main loop ═══════════════════════════════════════════════
rows=[]
for pid in PROJECTS:
    print(f"\n════ {pid} ════")
    base = checkout(pid)

    evo = Path(f"{base}_evo")
    if not (ARGS.resume_last and evo.exists()):
        evo = clone(base,"evo"); print("  🧬 EvoSuite…"); run_evosuite(evo)
    else: print("  🧬 EvoSuite reuse")
    rows.append(coverage(evo,"EvoSuite",pid))

    gpt = Path(f"{base}_gpt")
    if ARGS.resume_last and gpt.exists():
        print("  🤖 GPT resume…"); gpt_tests(gpt, True)
    else:
        gpt = clone(base,"gpt"); print("  🤖 GPT generate…"); gpt_tests(gpt, False)
    rows.append(coverage(gpt, GPT_MODEL, pid))

# ═══════════════ summary ════════════════════════════════════════════════
df=pd.DataFrame(rows)
df.to_csv(RUN_DIR/"coverage.csv", index=False)
print("\n📊  Coverage")
print(tabulate(df.pivot(index="project", columns="generator",
                        values=["line_%","branch_%"]).round(2),
               headers="keys", tablefmt="github"))
print(f"\n   {total_tokens} tokens used in GPT generation")

