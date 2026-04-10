# process_pisco_profiles.py

Standalone CLI tool for PISCO profile processing in Dash-app data pipelines.

It combines:
- segmentation (`run_segmenter`)
- metadata enrichment (CTD + templog)
- optional ViT-based classification
- EcoTaxa TSV export (always)
- optional EcoTaxa ZIP export

## Why this version is reusable

The script is now configurable and easier to move into a dedicated repository:
- Cruise/profile processing via CLI (`benchmark` or `cruise` mode)
- External JSON config (`--config`) for CTD paths, log paths, and model paths
- Model paths can also be overridden directly with CLI flags
- Cruise mode supports explicit profile lists:
  - `--profiles` (space and/or comma separated)
  - `--profiles-file` (newline-delimited file)

## Requirements

Core Python dependencies are already listed in the project `requirements.txt`.
The script needs segmentation + profile-analysis modules and supports either local files or pip packages:
- segmentation: local `segmenter.py` or package `pisco-segmenter`
- profile analysis: local `utils.py` (preferred), local `analyze_profiles_seavision.py` (legacy), or package `pisco-profile-utils`

Optional for model download from Hugging Face Hub:
- `huggingface_hub`

When moving to a new repo, either include local modules or install the two packages above.

## Quick start

### 1) Benchmark mode

```bash
python process_pisco_profiles.py \
  --mode benchmark \
  --source /mnt/filer \
  --output /data/pisco_benchmark_out \
  --profiles-per-cruise 5 \
  --export-zip
```

### 2) Cruise mode (all profiles)

```bash
python process_pisco_profiles.py \
  --mode cruise \
  --cruise SO298 \
  --source /mnt/filer \
  --output /data/pisco_cruise_out \
  --no-export-zip
```

### 3) Cruise mode (specific profiles from CLI)

```bash
python process_pisco_profiles.py \
  --mode cruise \
  --cruise SO298 \
  --output /data/pisco_cruise_out \
  --profiles SO298_001_20210615-120000,SO298_005_20210618-083000 SO298_011_20210622-090000
```

### 4) Cruise mode (specific profiles from file)

Create `profiles.txt`:

```text
# One profile per line
SO298_001_20210615-120000
SO298_005_20210618-083000
SO298_011_20210622-090000
```

Then run:

```bash
python process_pisco_profiles.py \
  --mode cruise \
  --cruise SO298 \
  --output /data/pisco_cruise_out \
  --profiles-file ./profiles.txt
```

## Config-driven workflow

Use `process_pisco_profiles.config.example.json` as template:

```bash
cp process_pisco_profiles.config.example.json process_pisco_profiles.config.json
# edit paths for your environment
```

Run with config:

```bash
python process_pisco_profiles.py \
  --mode cruise \
  --cruise SO298 \
  --output /data/pisco_cruise_out \
  --config ./process_pisco_profiles.config.json
```

CLI flags still override defaults where relevant (`--binary-model-dir`, `--living-model-dir`).

### Hugging Face model workflow (optional)

You can load models from Hugging Face Hub instead of fixed local paths:

```bash
python process_pisco_profiles.py \
  --mode cruise \
  --cruise SO298 \
  --output /data/pisco_cruise_out \
  --binary-model-hf your-org/pisco-vit-binary \
  --living-model-hf your-org/pisco-vit-living \
  --model-revision main
```

Equivalent settings can also be placed in `model_hub` inside the JSON config.

## End-to-end workflow (for Dash app suites)

1. **Select input scope**
   - `benchmark` for representative cross-cruise sets
   - `cruise` for full or targeted cruise runs
2. **Run segmentation + post-analysis**
   - outputs per profile under result folders (`Data`, `Deconv_crops`, `EcoTaxa`, `ViT_predictions.csv`)
3. **Feed outputs to Dash apps**
   - metadata CSVs and EcoTaxa TSVs are ready for downstream dashboards
4. **Optional publishing/export**
   - enable `--export-zip` for EcoTaxa upload bundles

## Preparing a new GitHub repository

Recommended minimal structure:

```text
pisco-profile-processor/
├── process_pisco_profiles.py
├── process_pisco_profiles.config.example.json
├── utils.py
├── requirements.txt
└── README.md
```

Suggested steps:
1. Copy the files above into the new repo
2. Rename this README to `README.md`
3. Replace path defaults in config example with project/team paths
4. Add CI check running:
   - `python -m py_compile process_pisco_profiles.py`
5. Tag first release once one cruise run succeeds end-to-end

## CLI reference

```bash
python process_pisco_profiles.py --help
```

Key options:
- `--mode {benchmark,cruise}`
- `--cruise <name>`
- `--source <path>`
- `--output <path>`
- `--profiles-per-cruise <int>`
- `--profile-limit <int>`
- `--profiles <list>`
- `--profiles-file <file>`
- `--export-zip | --no-export-zip`
- `--no-deconv`
- `--no-postanalysis`
- `--no-vit`
- `--config <json>`
- `--binary-model-dir <path>`
- `--living-model-dir <path>`
- `--binary-model-hf <repo-id>`
- `--living-model-hf <repo-id>`
- `--model-revision <revision>`
- `--model-cache-dir <path>`
