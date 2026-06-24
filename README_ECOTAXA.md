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
