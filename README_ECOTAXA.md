EcoTaxa upload helper
=====================

Script: `upload_to_ecotaxa.py`

Purpose: upload an already-created EcoTaxa ZIP and trigger an import job into a project via the EcoTaxa API + FTP.

## Quick start

```bash
python upload_to_ecotaxa.py /media/veit/T710_data/pisco_processed/ATAIR-BSH/20260423-1705/20260423-1705_Results/deconv_crops.zip \
    --project-id 22028
```

The script:
1. Uploads the ZIP to the EcoTaxa FTP server, preserving the local directory structure under a configurable subdirectory (e.g. `Ecotaxa_Data_to_import/GEOMAR/pisco_processed/...`).
2. Triggers an import job via the EcoTaxa REST API using the correct `source_path` (`FTP/Ecotaxa_Data_to_import/...`).

## Installation

```bash
pip install git+https://github.com/ecotaxa/ecotaxa_py_client.git
```

## Configuration

Copy `process_pisco_profiles.config.example.json` to your local config and fill in the credentials under the `ecotaxa` key:

```json
"ecotaxa": {
  "host": "https://ecotaxa.obs-vlfr.fr/api",
  "username": "you@example.com",
  "password": "yourpassword",
  "project_id": 22028,
  "ftp": {
    "host": "plankton.obs-vlfr.fr",
    "username": "ftp_user",
    "password": "ftp_pass",
    "remote_dir": "/plankton_rw/ftp_plankton/Ecotaxa_Data_to_import",
    "subdir": "GEOMAR",
    "local_root": "/media/veit/T710_data"
  }
}
```

Key FTP settings:

| Key | Description |
|-----|-------------|
| `remote_dir` | FTP path to the `Ecotaxa_Data_to_import` directory on the server |
| `subdir` | Subdirectory created under `Ecotaxa_Data_to_import` to organise uploads (e.g. `GEOMAR`) |
| `local_root` | Local path prefix stripped from the file path to derive the server-relative subpath. With the example above, `/media/veit/T710_data/pisco_processed/ATAIR-BSH/.../deconv_crops.zip` is uploaded to `Ecotaxa_Data_to_import/GEOMAR/pisco_processed/ATAIR-BSH/.../deconv_crops.zip`. |

All FTP settings can also be overridden on the command line (`--ftp-host`, `--ftp-user`, `--ftp-pass`, `--ftp-remote-dir`, `--ftp-subdir`, `--local-root`) or via environment variables `ECOTAXA_FTP_HOST`, `ECOTAXA_FTP_USER`, `ECOTAXA_FTP_PASS`.

## Updating existing objects

To re-import a ZIP and update metadata (e.g. after a pressure fix) without creating duplicates:

```bash
python upload_to_ecotaxa.py /path/to/deconv_crops.zip \
    --project-id 22028 \
    --update-mode Yes \
    --skip-existing-objects
```

## Exporting from the C++ segmenter (HDF5 route)

`export_ecotaxa_zips.py` above works on the **old Python pipeline's** output —
a `_Results` folder with `Crops/`, `Deconv_crops/` and a per-profile TSV. The
C++ segmenter produces none of that: crops, masks and bounding boxes live
inside `result.h5`. Use `export_hdf5_to_ecotaxa.py` for that route.

```bash
python export_hdf5_to_ecotaxa.py /path/to/segmenter/results/result.h5 \
    -o /path/to/export --profile-name M202_028-02
```

It writes one PNG per object, an `ecotaxa_<profile>.tsv` and a zip, in the same
shape the importer already accepts — so `upload_to_ecotaxa.py` takes the result
unchanged. Column names follow `rename_for_ecotaxa()`, so exports from both
routes share a schema.

The archive must come from `Mode::OBJECT_EXTRACTION` (`result.h5`), not from the
correction checkpoints. If `pisco_deconvolve` ran **before** extraction, the
crops are already deconvolved — deconvolution rewrites the correction files in
place and extraction reads those.

Morphometrics are computed from the segmenter's own `maskPixelData`, so unlike
the old route there is no need to re-threshold the saved crop to recover a mask.

Options worth knowing:

| Flag | Meaning |
|------|---------|
| `--profile-name` | EcoTaxa `sample_id` and archive name; defaults to the results directory name |
| `--max-zip-size` | split above this many MB; each part carries its own TSV |
| `--min-size` | drop objects whose bounding box is at or below this in both dimensions |
| `--gauge-pressure` | treat the pressure in the frame name as gauge; by default it is absolute and 10.1325 dbar is subtracted for depth |

### Linking crops back to the raw data

With `--link-base` (or the `linkBase` setting) every object gets an EcoTaxa
`object_link`. Clicking it in EcoTaxa's object details opens the HDF profile
viewer on that exact object, in its frame, with the detection boxes drawn — and
a *raw frame* toggle that swaps the reconstruction for the original image the
camera recorded.

```bash
python export_hdf5_to_ecotaxa.py /path/to/results -o /out \
    --link-base http://10.11.11.110:5000
```

The link is `<<base>>/v/<<workflow-id>>/<<frame>>/<<object number>>`, which the Hub
resolves to the right `result.h5` at click time. It names the workflow rather
than the file for two reasons: `object_link` is only `VARCHAR(255)` and a
path-based URL came to 242 characters, and a stored path would dead-link the
whole project if results ever moved. Links longer than the limit are dropped
with a warning rather than failing the row.

Outside a Hub workflow directory there is no workflow id to find, so pass
`--workflow-id` explicitly or leave the links off.

These URLs point at the Hub on the internal network — they are for people who
can reach it, not for collaborators on a public EcoTaxa instance.

### As a Hub tool

`toolbox.json` registers this as `pisco_ecotaxa_export`, so it can be the last
step of a Hub workflow:

```bash
hub providers add /opt/pisco/tools/process_pisco_profiles
```

Chain it after extraction, pointing `sourcePath` at the *correction* task's
output directory — extraction writes `result.h5` beside the checkpoints rather
than into its own output directory:

```json
{"tool": "pisco_ecotaxa_export",
 "settings": {"sourcePath": {"$from": {"task": 0, "output": "results"}},
              "targetPath": "", "profileName": "M202_028-02"}}
```

## Re-exporting EcoTaxa ZIPs

If you need to regenerate ZIPs from already-processed profiles (e.g. after a metadata fix), use `export_ecotaxa_zips.py`:

```bash
# Single profile
python export_ecotaxa_zips.py /path/to/20260423-1705_Results

# Multiple profiles or a whole cruise folder
python export_ecotaxa_zips.py \
    /media/veit/T710_data/pisco_processed/ATAIR-BSH/20260423-1705/20260423-1705_Results \
    /media/veit/T710_data/pisco_processed/ATAIR-BSH/20260424-0900/20260424-0900_Results

# Or pass the cruise root to process all profiles at once
python export_ecotaxa_zips.py /media/veit/T710_data/pisco_processed/ATAIR-BSH
```
