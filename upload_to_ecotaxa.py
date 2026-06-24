#!/usr/bin/env python3
"""Upload a file (zip or directory archive) to EcoTaxa and trigger import.

Uses the `ecotaxa_py_client` generated client. Install with:
  pip install git+https://github.com/ecotaxa/ecotaxa_py_client.git

Basic flow:
 - login
 - upload file to user area (`FilesApi.post_user_file`)
 - call `ProjectsApi.import_file` with returned server path
"""
import argparse
import base64
import ftplib
import json
import os
import mimetypes
import urllib.error
import urllib.request
import urllib.parse
import sys
from pathlib import Path


def load_config(path):
    with open(path, "r") as fh:
        return json.load(fh)


def extract_token(login_result):
    if isinstance(login_result, str):
        return login_result
    for attr in ("access_token", "token", "accessToken"):
        value = getattr(login_result, attr, None)
        if value:
            return value
    if isinstance(login_result, dict):
        for key in ("access_token", "token", "accessToken"):
            value = login_result.get(key)
            if value:
                return value
    return login_result


def multipart_upload(url, token, file_path, remote_path, tag=None):
    boundary = "----EcoTaxaBoundary7d1e0b7c"
    filename = file_path.name
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts = []

    def add_field(name, value):
        parts.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            str(value).encode(),
            b"\r\n",
        ])

    def add_file(name, path_obj):
        parts.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"; filename="{path_obj.name}"\r\n'.encode(),
            f"Content-Type: {mime_type}\r\n\r\n".encode(),
            path_obj.read_bytes(),
            b"\r\n",
        ])

    add_file("file", file_path)
    add_field("path", remote_path)
    if tag is not None:
        add_field("tag", tag)
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))

    with urllib.request.urlopen(req) as resp:
        payload = resp.read().decode("utf-8", errors="replace").strip()
    try:
        return json.loads(payload)
    except Exception:
        return payload


def _tus_metadata_value(value):
    return base64.b64encode(str(value).encode("utf-8")).decode("ascii")


def tus_upload(base_url, token, file_path, remote_path, tag=None, chunk_size=8 * 1024 * 1024):
    upload_url = f"{base_url.rstrip('/')}/user_files/upload/"
    upload_size = file_path.stat().st_size
    filename = file_path.name

    create_req = urllib.request.Request(upload_url, data=b"", method="POST")
    create_req.add_header("Authorization", f"Bearer {token}")
    create_req.add_header("Tus-Resumable", "1.0.0")
    create_req.add_header("Upload-Length", str(upload_size))
    create_req.add_header(
        "Upload-Metadata",
        ",".join(
            [
                f"filename {_tus_metadata_value(filename)}",
                f"path {_tus_metadata_value(remote_path)}",
                f"source_path {_tus_metadata_value(remote_path)}",
                *( [f"tag {_tus_metadata_value(tag)}"] if tag is not None else [] ),
            ]
        ),
    )

    with urllib.request.urlopen(create_req) as resp:
        location = resp.headers.get("Location") or resp.headers.get("location")
        body = resp.read().decode("utf-8", errors="replace").strip()

    if location:
        upload_url = urllib.parse.urljoin(upload_url, location)
    elif body:
        upload_url = body

    offset = 0
    with file_path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break

            patch_req = urllib.request.Request(upload_url, data=chunk, method="PATCH")
            patch_req.add_header("Authorization", f"Bearer {token}")
            patch_req.add_header("Tus-Resumable", "1.0.0")
            patch_req.add_header("Upload-Offset", str(offset))
            patch_req.add_header("Content-Type", "application/offset+octet-stream")
            patch_req.add_header("Content-Length", str(len(chunk)))

            with urllib.request.urlopen(patch_req) as resp:
                new_offset = resp.headers.get("Upload-Offset")
                if new_offset is not None:
                    offset = int(new_offset)
                else:
                    offset += len(chunk)

    return upload_url


def _ftp_mkdirs(ftp, rel_dir):
    """Navigate into rel_dir from current position, creating parts as needed."""
    for part in [p for p in rel_dir.split("/") if p]:
        try:
            ftp.cwd(part)
        except ftplib.all_errors:
            ftp.mkd(part)
            ftp.cwd(part)


def ftp_upload(host, username, password, file_path, remote_dir, remote_name=None):
    """Upload file_path to remote_dir on the FTP server.

    remote_dir may be an absolute server path (e.g. /plankton_rw/ftp_plankton/Ecotaxa_Data_to_import/...)
    or a relative path. After login the FTP home is used as the base; if remote_dir starts with
    the home prefix it is stripped so navigation stays relative to home throughout.
    Returns the absolute FTP path of the uploaded file.
    """
    filename = remote_name or file_path.name
    with ftplib.FTP(host) as ftp:
        ftp.login(username, password)
        ftp_home = ftp.pwd().rstrip("/")

        # Compute relative path from FTP home so we never navigate to absolute
        # server paths that are inaccessible via FTP (e.g. /plankton_rw/...).
        remote_dir = (remote_dir or "").rstrip("/")
        if ftp_home and remote_dir.startswith(ftp_home + "/"):
            rel_dir = remote_dir[len(ftp_home) + 1:]
        else:
            rel_dir = remote_dir.lstrip("/")

        _ftp_mkdirs(ftp, rel_dir)
        with file_path.open("rb") as fh:
            ftp.storbinary(f"STOR {filename}", fh)
        actual_dir = ftp.pwd().rstrip("/")
        return f"{actual_dir}/{filename}"


def main():
    p = argparse.ArgumentParser(description="Upload and import file to EcoTaxa")
    p.add_argument("file", help="Local file to upload (zip or archive)")
    p.add_argument("--config", default="process_pisco_profiles.config.example.json", help="Config JSON path")
    p.add_argument("--host", help="Override EcoTaxa API host (e.g. https://ecotaxa.obs-vlfr.fr/api)")
    p.add_argument("--project-id", type=int, help="Project id to import into (overrides config)")
    p.add_argument("--tag", help="Tag to group uploaded files on server", default=None)
    p.add_argument("--ftp-host", help="Override EcoTaxa FTP host")
    p.add_argument("--ftp-user", help="Override EcoTaxa FTP username")
    p.add_argument("--ftp-pass", help="Override EcoTaxa FTP password")
    p.add_argument("--ftp-remote-dir", help="Override EcoTaxa FTP upload root (e.g. /plankton_rw/ftp_plankton/Ecotaxa_Data_to_import)")
    p.add_argument("--ftp-subdir", help="Subdirectory under the FTP import root to organise uploads (e.g. GEOMAR)")
    p.add_argument("--local-root", help="Local path prefix to strip when building the server path (e.g. /media/veit/T710_data)")
    p.add_argument("--skip-loaded-files", action="store_true", help="Set import.skip_loaded_files=True")
    p.add_argument("--skip-existing-objects", action="store_true", help="Set import.skip_existing_objects=True")
    p.add_argument("--update-mode", default="", help="Import update mode ('', 'Yes', 'Cla')")
    p.add_argument("--no-verify-ssl", action="store_true", help="Disable SSL verification for API client")
    args = p.parse_args()

    cfg = load_config(args.config)
    ecfg = cfg.get("ecotaxa", {})
    ftpcfg = ecfg.get("ftp", {})
    host = args.host or ecfg.get("host", "https://ecotaxa.obs-vlfr.fr/api")
    username = ecfg.get("username")
    password = ecfg.get("password")
    project_id = args.project_id or ecfg.get("project_id")
    ftp_host = args.ftp_host or ftpcfg.get("host") or os.environ.get("ECOTAXA_FTP_HOST") or "plankton.obs-vlfr.fr"
    ftp_user = args.ftp_user or ftpcfg.get("username") or os.environ.get("ECOTAXA_FTP_USER")
    ftp_pass = args.ftp_pass or ftpcfg.get("password") or os.environ.get("ECOTAXA_FTP_PASS")
    ftp_remote_dir = args.ftp_remote_dir or ftpcfg.get("remote_dir") or "/plankton_rw/ftp_plankton/Ecotaxa_Data_to_import"
    ftp_subdir = args.ftp_subdir or ftpcfg.get("subdir") or ""
    local_root = args.local_root or ftpcfg.get("local_root") or ""

    if not username or not password:
        # allow reading password from env or prompt
        username = username or os.environ.get("ECOTAXA_USER")
        password = password or os.environ.get("ECOTAXA_PASS")

    if not username:
        print("Missing Ecotaxa username (set in config or ECOTAXA_USER)")
        sys.exit(1)
    if not password:
        # avoid putting credentials in repo; prompt if empty
        try:
            import getpass

            password = getpass.getpass(prompt=f"Password for {username}: ")
        except Exception:
            print("Missing Ecotaxa password (set in config, ECOTAXA_PASS env, or enter interactively)")
            sys.exit(1)

    if not project_id:
        print("Missing target project id (set in config or pass --project-id)")
        sys.exit(1)

    if ftp_host:
        if not ftp_user:
            ftp_user = username
        if not ftp_pass:
            try:
                import getpass

                ftp_user_prompt = ftp_user or username
                ftp_user = input(f"FTP username [{ftp_user_prompt}]: ").strip() or ftp_user_prompt
                ftp_pass = getpass.getpass(prompt=f"FTP password for {ftp_user}: ")
            except Exception:
                print("Missing FTP credentials (set ecotaxa.ftp.* in config or ECOTAXA_FTP_* env vars)")
                sys.exit(1)

    try:
        import ecotaxa_py_client
        from ecotaxa_py_client.rest import ApiException
    except ImportError:
        print("ecotaxa_py_client is not installed. Install with:\n  pip install git+https://github.com/ecotaxa/ecotaxa_py_client.git")
        sys.exit(2)

    configuration = ecotaxa_py_client.Configuration(host=host)
    if args.no_verify_ssl:
        configuration.verify_ssl = False

    # login
    try:
        with ecotaxa_py_client.ApiClient(configuration) as api_client:
            auth_api = ecotaxa_py_client.AuthentificationApi(api_client)
            login_req = ecotaxa_py_client.LoginReq(username=username, password=password)
            token = extract_token(auth_api.login(login_req))
            configuration.access_token = token
    except ApiException as e:
        print("Login failed:", e)
        sys.exit(3)

    # upload file
    filename = Path(args.file)
    if not filename.exists():
        print(f"File not found: {filename}")
        sys.exit(1)

    # Build the server-relative import path and the corresponding FTP upload directory.
    # ftp_remote_dir is the FTP path to "Ecotaxa_Data_to_import"; its last component is
    # reused as the root of the EcoTaxa-visible import path.
    ftp_import_root_name = Path(ftp_remote_dir).name  # e.g. "Ecotaxa_Data_to_import"
    if local_root:
        try:
            rel_path = filename.resolve().relative_to(Path(local_root).resolve())
        except ValueError:
            print(f"Warning: file {filename} is not under local_root {local_root}; using filename only")
            rel_path = Path(filename.name)
    else:
        rel_path = Path(filename.name)

    # import_source_path: what EcoTaxa receives as source_path.
    # EcoTaxa expects FTP-uploaded files to be prefixed with "FTP/", e.g.
    #   FTP/Ecotaxa_Data_to_import/GEOMAR/pisco_processed/.../deconv_crops.zip
    subdir_parts = [p for p in [ftp_subdir] if p]
    import_source_path = "/".join(["FTP", ftp_import_root_name] + subdir_parts + [str(rel_path)])

    # ftp_rel_dir: relative path from FTP home for navigation, e.g.
    #   Ecotaxa_Data_to_import/GEOMAR/pisco_processed/.../Results
    # This avoids passing absolute server paths (/plankton_rw/...) that the FTP
    # client cannot navigate to.
    if local_root and rel_path.parent != Path("."):
        ftp_rel_dir = "/".join([ftp_import_root_name] + subdir_parts + [str(rel_path.parent)])
    else:
        ftp_rel_dir = "/".join([ftp_import_root_name] + subdir_parts)

    server_path = None
    if ftp_host and ftp_user and ftp_pass:
        try:
            ftp_actual = ftp_upload(ftp_host, ftp_user, ftp_pass, filename, ftp_rel_dir, remote_name=filename.name)
            # ftp_actual is an absolute FTP path; strip the home prefix to get the
            # Derive EcoTaxa source_path: strip any FTP home prefix before
            # Ecotaxa_Data_to_import, then prepend the required "FTP/" prefix.
            actual = ftp_actual.lstrip("/")
            marker = ftp_import_root_name + "/"
            if marker in actual:
                actual = actual[actual.index(marker):]
            server_path = "FTP/" + actual
            print(f"Uploaded via FTP {ftp_host}; FTP path: {ftp_actual}; import source_path: {server_path}")
            if server_path != import_source_path:
                print(f"Note: actual upload path differs from intended ({import_source_path})")
        except Exception as ftp_err:
            print("FTP upload failed, falling back to EcoTaxa API upload paths:", ftp_err)

    with ecotaxa_py_client.ApiClient(configuration) as api_client:
        files_api = ecotaxa_py_client.FilesApi(api_client)
        if server_path is None:
            try:
                # Prefer the TUS upload path for large archives.
                server_path = tus_upload(host, token, filename, str(filename), tag=args.tag)
                print("Uploaded via TUS to server path:", server_path)
            except (ApiException, urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
                print("TUS upload failed, trying EcoTaxa client upload:", e)

                try:
                    # pass the local filename path so the generated client can open it
                    server_path = files_api.post_user_file(file=str(filename), path=str(filename), tag=args.tag)
                    print("Uploaded to server path:", server_path)
                except ApiException as client_err:
                    if getattr(client_err, "status", None) != 404:
                        print("File upload failed:", client_err)
                        sys.exit(4)

                    # Fallback for the live EcoTaxa API, which exposes upload as /user_files/.
                    base = host.rstrip("/")
                    token = configuration.access_token
                    remote_name = str(filename)
                    attempted = []
                    server_path = None
                    for upload_url in (f"{base}/user_files/", f"{base}/user_files", f"{base}/my_files/", f"{base}/my_files"):
                        attempted.append(upload_url)
                        try:
                            server_path = multipart_upload(upload_url, token, filename, remote_name, tag=args.tag)
                            print(f"Uploaded via fallback URL {upload_url}:", server_path)
                            break
                        except urllib.error.HTTPError as http_err:
                            if http_err.code != 404:
                                raise
                        except urllib.error.URLError:
                            raise

                    if server_path is None:
                        print("File upload failed: 404 from EcoTaxa upload endpoint")
                        print("Tried:")
                        for url in attempted:
                            print(" -", url)
                        print("FTP fallback was tried first" if (ftp_host and ftp_user and ftp_pass) else "FTP fallback not configured")
                        sys.exit(4)

        # prepare import request
        import_req = ecotaxa_py_client.ImportReq(
            source_path=server_path,
            skip_loaded_files=bool(args.skip_loaded_files),
            skip_existing_objects=bool(args.skip_existing_objects),
            update_mode=args.update_mode,
        )

        projects_api = ecotaxa_py_client.ProjectsApi(api_client)
        try:
            print(f"Triggering import into project {project_id}...")
            rsp = projects_api.import_file(project_id, import_req)
            print("Import response:")
            print(rsp)
        except ApiException as e:
            print("Import request failed:", e)
            sys.exit(5)


if __name__ == "__main__":
    main()
