## 0  Create & activate the virtual-env

```bash
# inside the project root
python3 -m venv .venv          # create
# If in Windows:
.venv\Scripts\activate
# ---
# If in macOS / Linux:
source .venv/bin/activate
# ---

python -m pip install --upgrade pip wheel
```

*(after this every `python` / `pip` you call is safely scoped to `.venv`)*

## 1  Install deps

```bash
pip install -r requirements.txt
```

## 2  Setup the dataset
```bash
mkdir data
cd data
git clone https://github.com/rjust/defects4j
cd defects4j
cpanm --installdeps .
./init.sh
```


## 3  Run the benchmark

### IMPORTANT:
Please go to the .env file and fill in your OpenAI API Key first.

Also, before running, please go to  test_generation_pipeline.py and check or adjust the configuration in line 35.

### Please run 
```bash
python3 test_generation_pipeline.py
```

## 4  Deactivate when you’re done

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

That’s it, Happy testing! 🚀
