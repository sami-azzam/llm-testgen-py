## 0  Create & activate the virtual-env

```bash
# inside the project root
python3 -m venv .venv          # create
source .venv/bin/activate      # activate (fish: source .venv/bin/activate.fish; Windows: .venv\Scripts\activate)
python -m pip install --upgrade pip wheel
```

*(after this every `python` / `pip` you call is safely scoped to `.venv`)*

## 1  Install deps

```bash
pip install -r requirements.txt
```

## 2  Run the benchmark

```bash
export OPENAI_API_KEY="<your-key>"
export PATH=$PATH:$HOME/defects4j/framework/bin     # once per shell
python test_generation_pipeline.py
```

## 3  Deactivate when you’re done

```bash
deactivate
```

---

### Optional one-liner setup script

Save as `setup_env.sh`, `chmod +x setup_env.sh`, then run `./setup_env.sh`:

```bash
#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
echo "✔  venv ready.  Run:  source .venv/bin/activate"
```

Now your whole Defects4J + EvoSuite + GPT tool-chain lives inside `.venv` and can be nuked with a simple `rm -rf .venv` if you ever want a clean slate.

That’s it, man. 🚀
