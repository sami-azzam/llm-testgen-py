#!/usr/bin/env python3
"""
test_generation_pipeline_planA.py
────────────────────────────────────────────────────────────────────────────
Benchmark Defects4J bugs with

  • EvoSuite          (search-based, Java 8 mode)
  • GPT (LLM-based, user-supplied prompt)

Plan A patches each checked-out project to compile with –source 8 –target 1.8
so modern Java syntax from EvoSuite / GPT compiles cleanly.

The script is resumable ( --resume-last ), quarantines failing GPT tests,
and prints JaCoCo line / branch coverage.
"""

# ───────────────────────── std-lib ────────────────────────────────────────
from __future__ import annotations
import os, subprocess, random, shutil, asyncio, re, textwrap, xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Tuple

# ─────────────────────── 3rd-party deps ───────────────────────────────────
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tabulate import tabulate
from tqdm import tqdm

# ─────────────────────── local helper ─────────────────────────────────────
from java_finder import find_java11                    # user-supplied util

# ═════════════════════════ configuration ══════════════════════════════════
SEED                 = 2025
NUM_PROJECTS         = 1
BUG_REV              = "1f"
GPT_MODEL            = "gpt-4o-mini"                   # example; keep your key
MAX_PAR              = 100                             # parallel GPT calls
EVOSUITE_MEM_MB      = 4096

ROOT          = Path(__file__).resolve().parents[1]
RESULTS_DIR   = ROOT / "results"
EVOSUITE_JAR  = ROOT / "src/tools/evosuite.jar"

# ═════════════════════ env / paths ════════════════════════════════════════
load_dotenv(ROOT / ".env")
D4J_BIN = Path(os.getenv("D4J_BIN", ROOT / "data/defects4j/framework/bin")).expanduser().resolve()
os.environ["PATH"] = f"{D4J_BIN}:{os.environ['PATH']}"

ENV = {**os.environ,
       "JAVA_HOME": find_java11(),          # any JDK 11+ fine for source 8
       "D4J_COVERAGE": "jacoco",
       "TZ": "America/Los_Angeles"}

client = AsyncOpenAI()                      # needs OPENAI_API_KEY

# ═════════════════════ utilities ═════════════════════════════════════════
def sh(cmd: List[str], cwd: Path | None = None, hide: bool=False,
       cap: bool=False) -> Tuple[int,str,str] | None:
    if not hide:
        print(" ".join(cmd))
    if cap:
        p = subprocess.run(cmd, cwd=cwd, env=ENV, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode, p.stdout, p.stderr
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True,
                   stdout=subprocess.DEVNULL if hide else None,
                   stderr=subprocess.STDOUT if hide else None)

# ═════════════ project sampling ══════════════════════════════════════════
ALL = ["Chart","Cli","Closure","Codec","Collections","Compress","Csv","Gson",
       "JacksonCore","JacksonDatabind","JacksonXml","Jsoup","JxPath","Lang",
       "Math","Mockito","Time"]
random.Random(SEED).shuffle(ALL)
PROJECTS = ALL[:NUM_PROJECTS]

# ═════════════ argument handling ═════════════════════════════════════════
ap = argparse.ArgumentParser()
ap.add_argument("--resume-last", action="store_true")
ARGS = ap.parse_args()

runs = sorted(RESULTS_DIR.glob("runs_*"))
RUN_DIR = runs[-1] if (ARGS.resume_last and runs) else \
          RESULTS_DIR / f"runs_{datetime.now():%Y%m%d_%H%M%S}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ═════════════ helper: patch build to Java 8 ═════════════════════════════
def relax_source_target(work_dir: Path) -> None:
    """
    Replace source/target 6 → 8 in every Ant / Maven build file under the
    checkout.  Executed once per project.
    """
    for f in work_dir.rglob("*.xml"):
        txt = f.read_text()
        if 'source="6"' in txt or 'target="1.6"' in txt:
            txt = re.sub(r'source="6"', 'source="8"', txt)
            txt = re.sub(r'target="1\.6"', 'target="1.8"', txt)
            f.write_text(txt)

# ═════════════ Defects4J wrappers ════════════════════════════════════════
def checkout(pid: str) -> Path:
    base = RUN_DIR / f"{pid}_base"
    if not base.exists():
        sh(["defects4j","checkout","-p",pid,"-v",BUG_REV,"-w",base], hide=True)
        relax_source_target(base)
    return base

def clone(src: Path, suf: str) -> Path:
    dst = Path(f"{src}_{suf}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst

# ═════════════ EvoSuite generation (Java 8) ══════════════════════════════
def evosuite(work: Path) -> None:
    cp = subprocess.check_output(
        ["defects4j","export","-p","cp.compile"], cwd=work, env=ENV, text=True).strip()
    tdir = work / subprocess.check_output(
        ["defects4j","export","-p","dir.src.tests"], cwd=work, env=ENV, text=True).strip()
    tdir.mkdir(parents=True, exist_ok=True)

    rc, out, err = sh(
        ["java", f"-Xmx{EVOSUITE_MEM_MB}m", "-jar", str(EVOSUITE_JAR),
         "-projectCP", cp,
         "-generateSuite", "-class", ".*",
         "-Dtest_dir", tdir],
        cwd=work, cap=True, hide=True)
    if rc:
        (work/"evosuite_err.log").write_text(out+err)
        print("   ⚠️  EvoSuite failed (see evosuite_err.log)")

    # ── move *ESTest.java → *Test.java so Ant includes them ──────────────
    for src in tdir.rglob("*ESTest*.java"):
        dst = src.with_name(src.name.replace("ESTest", "Test"))
        src.rename(dst)

# ═════════════ GPT generation  (prompt blank) ════════════════════════════
def sanitize_name(fqcn: str, code: str) -> str:
    simple = fqcn.split('.')[-1] + "GptTest"
    # replace first 'class XYZ' with correct name
    return re.sub(r'\bclass\s+\w+', f'class {simple}', code, count=1)

def gpt_tests(work: Path, skip_existing: bool) -> None:
    rel_tests = subprocess.check_output(
        ["defects4j","export","-p","dir.src.tests"], cwd=work, env=ENV, text=True).strip()
    test_dir = work / rel_tests; test_dir.mkdir(parents=True, exist_ok=True)

    classes = subprocess.check_output(
        ["defects4j","export","-p","classes.relevant"], cwd=work, env=ENV, text=True).splitlines()
    todo = [(c, test_dir/f"{c.split('.')[-1]}GptTest.java")
            for c in classes if not (skip_existing and (test_dir/f"{c.split('.')[-1]}GptTest.java").exists())]
    if not todo:
        print("   ⏩  all GPT tests present"); return

    # locate source root and map
    src_root = work / subprocess.check_output(
        ["defects4j","export","-p","dir.src.classes"], cwd=work, env=ENV, text=True).strip()
    fqcn2file = {c: src_root/Path(c.replace(".","/")+".java") for c in classes}

    async def _one(fqcn: str, sem: asyncio.Semaphore):
        path = fqcn2file.get(fqcn)
        if not path.is_file(): return None
        source = path.read_text()

        prompt = f"""
Here is the *exact* source code of `{fqcn}`:

```java
{source}
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
        async with sem:
            rsp = await client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role":"user","content":prompt}],
            )
        code = rsp.choices[0].message.content
        code = re.sub(r'^```(?:java)?\s*','',code)
        code = re.sub(r'\s*```$','',code, flags=re.S)
        code = sanitize_name(fqcn, textwrap.dedent(code))
        return fqcn, code

    async def run_all():
        sem = asyncio.Semaphore(MAX_PAR)
        coros = [_one(fqcn, sem) for fqcn,_ in todo]
        pbar = tqdm(total=len(todo), desc="   GPT", unit="test")
        for fut in asyncio.as_completed(coros):
            res = await fut
            if res:
                fqcn, src = res
                (test_dir/f"{fqcn.split('.')[-1]}GptTest.java").write_text(src)
            pbar.update(1)
        pbar.close()
    asyncio.run(run_all())

# ═════════════ prune non-compiling GPT tests (now Java 8) ═════════════════
def prune_bad(work: Path):
    cp_compile = subprocess.check_output(
        ["defects4j","export","-p","cp.compile"], cwd=work, env=ENV, text=True).strip()
    test_src = work / subprocess.check_output(
        ["defects4j","export","-p","dir.src.tests"], cwd=work, env=ENV, text=True).strip()
    for f in test_src.rglob("*GptTest.java"):
        rc = subprocess.run(["javac","-classpath",cp_compile,str(f)],
                            cwd=work, env=ENV,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if rc: f.rename(f.with_suffix(".java.disabled"))

# ═════════════ coverage helper (recursive counters) ══════════════════════
def coverage(work: Path, tag: str, proj: str) -> dict:
    if tag.startswith("GPT"):
        prune_bad(work)

    comp_rc = subprocess.run(["defects4j","compile"], cwd=work, env=ENV).returncode
    if comp_rc: return {"project":proj,"generator":tag,"line_%":0,"branch_%":0}

    subprocess.run(["defects4j","test"],      cwd=work, env=ENV, stdout=subprocess.DEVNULL)

    subprocess.run(["defects4j","coverage", "-t", "jacoco"],  cwd=work, env=ENV, stdout=subprocess.DEVNULL)

    xml = work/"coverage.xml"
    if not xml.is_file(): return {"project":proj,"generator":tag,"line_%":0,"branch_%":0}
    root = ET.parse(xml).getroot()
    lines   = [int(c.get('covered')) for c in root.iterfind(".//counter[@type='LINE']")]
    missed  = [int(c.get('missed' )) for c in root.iterfind(".//counter[@type='LINE']")]
    branch  = [int(c.get('covered')) for c in root.iterfind(".//counter[@type='BRANCH']")]
    bmiss   = [int(c.get('missed' )) for c in root.iterfind(".//counter[@type='BRANCH']")]
    lpct = round(100*sum(lines)/(sum(lines)+sum(missed)),2) if lines else 0
    bpct = round(100*sum(branch)/(sum(branch)+sum(bmiss)),2) if branch else 0
    return {"project":proj,"generator":tag,"line_%":lpct,"branch_%":bpct}

# ═════════════ main loop ═════════════════════════════════════════════════
results = []

for pid in PROJECTS:
    print(f"\n════ {pid} ════")
    base = checkout(pid)

    # EvoSuite
    evo = Path(f"{base}_evo")
    if not (ARGS.resume_last and evo.exists()):
        evo = clone(base,"evo"); print("  🧬 EvoSuite…"); evosuite(evo)
    else:          print("  🧬 EvoSuite reuse")
    results.append(coverage(evo, "EvoSuite", pid))

    # GPT
    gpt = Path(f"{base}_gpt")
    if ARGS.resume_last and gpt.exists():
        print("  🤖 GPT resume…"); gpt_tests(gpt, skip_existing=True)
    else:
        gpt = clone(base,"gpt"); print("  🤖 GPT generate…"); gpt_tests(gpt, False)
    results.append(coverage(gpt, GPT_MODEL, pid))

# ═════════════ summary ═══════════════════════════════════════════════════
df = pd.DataFrame(results)
df.to_csv(RUN_DIR/"coverage.csv", index=False)
print("\n📊  Coverage")
print(tabulate(df.pivot(index="project", columns="generator",
                        values=["line_%","branch_%"]).round(2),
               headers="keys", tablefmt="github"))
