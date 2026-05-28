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

### 5) Cruise mode (custom folders - single folder)

For new/uncurated cruises or non-standard folder structures, process a custom folder directly:

```bash
python process_pisco_profiles.py \
  --mode cruise \
  --cruise MY_NEW_CRUISE \
  --folder /mnt/data/202604_ATAIR-BSB/20260422-2051 \
  --output /data/pisco_custom_out
```

### 6) Cruise mode (custom folders - multiple folders via CLI)

Process multiple folders as separate profiles:

```bash
python process_pisco_profiles.py \
  --mode cruise \
  --cruise MY_NEW_CRUISE \
  --folders /mnt/data/folder1 /mnt/data/folder2 /mnt/data/folder3 \
  --output /data/pisco_custom_out
```

### 7) Cruise mode (custom folders - multiple folders from file)

Create `folders.txt`:

```text
# One folder path per line
/mnt/data/202604_ATAIR-BSB/20260422-2051
/mnt/data/202604_ATAIR-BSB/20260423-1345
/mnt/data/202605_ATAIR-BSB/20260501-0900
```

Then run:

```bash
python process_pisco_profiles.py \
  --mode cruise \
  --cruise MY_NEW_CRUISE \
  --folders-file ./folders.txt \
  --output /data/pisco_custom_out
```

Note: `--cruise` accepts any name you choose—it's used for metadata and logging. For uncurated/new cruises without standard directory structure, use custom names like `MY_NEW_CRUISE`, `ATAIR-BSB`, or similar. CTD and log files are optional; processing will proceed without them if not available.

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

## Abundance + biovolume post-processing

Use `calc_abundance_biovolume_pisco.py` after profile processing to aggregate class-wise
abundance and biovolume by depth bins, plus summary plots.

Basic usage:

```bash
python calc_abundance_biovolume_pisco.py /path/to/results_root
```

Write all CSV + PNG outputs into one folder:

```bash
python calc_abundance_biovolume_pisco.py \
  /path/to/results_root \
  --output-dir /path/to/all_outputs
```

Depth binning options:

```bash
# explicit edges
python calc_abundance_biovolume_pisco.py /path/to/results_root \
  --output-dir /path/to/all_outputs \
  --depth-bins 0,50,100,200,500,1000

# evenly spaced bins
python calc_abundance_biovolume_pisco.py /path/to/results_root \
  --output-dir /path/to/all_outputs \
  --depth-bin-step 10 \
  --depth-bin-max 2000
```

Output files are written as profile-prefixed files (for example
`<profile>_abundance_biovolume_by_depth.csv` and `<profile>_abundance_biovolume_by_depth.png`),
and cruise-level stacked plots are also written to `--output-dir`.

## Session updates (2026-05-28)

The following changes were implemented and validated during this session:

- **Custom folder processing in `process_pisco_profiles.py`**
  - Added `--folder` for a single direct image folder.
  - Added `--folders` for multiple folder paths via CLI.
  - Added `--folders-file` for newline/comma-delimited folder lists.
  - Cruise mode now supports non-standard/new cruise layouts without requiring the classic cruise directory structure.

- **Post-analysis robustness improvements**
  - Missing coordinates no longer crash processing (`Coordinates: N/A` is logged when metadata is absent).
  - `gen_crop_df` parsing in `utils.py` now supports additional filename layouts used in new deployments, including:
    - `date-time_pressure_temperature_index` (4-part)
    - `date-time_pressure_temperature` (3-part)
  - Fallback `date-time` handling prevents sorting crashes when date-time fields are missing from parsed columns.

- **Abundance/biovolume compatibility fixes (`calc_abundance_biovolume_pisco.py`)**
  - Added support for source images in `.tif`/`.tiff` (in addition to `.png`).
  - Pressure parsing now handles unit-suffixed strings (e.g. `000.930bar`) and converts consistently to dbar.
  - Improved depth-bin split handling to avoid shape mismatch errors on edge-case labels.
  - Added custom cruise detection from source paths like `.../202604_ATAIR-BSB_PISCO/...` so outputs show `ATAIR-BSB` instead of `UNKNOWN`.

- **Standalone ZIP export flow added**
  - Added `export_ecotaxa_zips.py` to create EcoTaxa ZIPs from existing results without rerunning segmentation/post-analysis.
  - Fixed EcoTaxa TSV loading to preserve 2-row MultiIndex headers expected by `create_ecotaxa_zips`.
  - Added folder filtering so only real profile result folders are exported.

Example ZIP export command:

```bash
python export_ecotaxa_zips.py /media/veit/T710_data/pisco_processed/ATAIR-BSH --max-zip-size 1000
```

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
- `--cruise <name>` (required for cruise mode; accepts any custom name)
- `--source <path>`
- `--output <path>` (required)
- `--profiles-per-cruise <int>`
- `--profile-limit <int>`
- `--profiles <list>` (space and/or comma-separated)
- `--profiles-file <file>` (newline-delimited file)
- `--folder <path>` (single custom folder, bypasses cruise discovery)
- `--folders <list>` (multiple custom folders, space-separated)
- `--folders-file <file>` (newline-delimited file with folder paths)
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
