#!/usr/bin/env python3
"""Parse RPM repodata and emit a packages.json manifest for the UI."""

import argparse
import gzip
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

RPM_NS = "http://linux.duke.edu/metadata/common"
OTHER_NS = "http://linux.duke.edu/metadata/other"
REPO_NS = "http://linux.duke.edu/metadata/repo"


def find_repodata_file(repodata_dir, data_type):
    """Read repomd.xml and return the path to a specific data type's file."""
    repomd_path = os.path.join(repodata_dir, "repomd.xml")
    tree = ET.parse(repomd_path)
    root = tree.getroot()

    for data in root.findall(f"{{{REPO_NS}}}data"):
        if data.get("type") == data_type:
            location = data.find(f"{{{REPO_NS}}}location")
            if location is not None:
                href = location.get("href", "")
                return os.path.join(os.path.dirname(repodata_dir), href)

    return None


def open_compressed(path):
    """Open a gzip or zstd compressed file for reading."""
    if path.endswith(".zst"):
        result = subprocess.run(
            ["zstdcat", path], capture_output=True, check=True
        )
        import io
        return io.BytesIO(result.stdout)
    else:
        return gzip.open(path, "rb")


def parse_primary(repodata_dir):
    """Parse primary.xml.{gz,zst} and return package metadata."""
    path = find_repodata_file(repodata_dir, "primary")
    if not path:
        print("error: primary metadata not found in repomd.xml", file=sys.stderr)
        sys.exit(1)

    packages = {}
    with open_compressed(path) as f:
        tree = ET.parse(f)

    for pkg in tree.getroot().findall(f"{{{RPM_NS}}}package"):
        if pkg.get("type") != "rpm":
            continue

        name = pkg.findtext(f"{{{RPM_NS}}}name", "")
        version_el = pkg.find(f"{{{RPM_NS}}}version")
        ver = version_el.get("ver", "") if version_el is not None else ""
        rel = version_el.get("rel", "") if version_el is not None else ""
        arch = pkg.findtext(f"{{{RPM_NS}}}arch", "")

        size_el = pkg.find(f"{{{RPM_NS}}}size")
        size = int(size_el.get("package", "0")) if size_el is not None else 0

        time_el = pkg.find(f"{{{RPM_NS}}}time")
        build_time = int(time_el.get("build", "0")) if time_el is not None else 0

        location_el = pkg.find(f"{{{RPM_NS}}}location")
        filename = os.path.basename(location_el.get("href", "")) if location_el is not None else ""

        key = f"{name}-{ver}-{rel}"
        packages[key] = {
            "name": name,
            "version": ver,
            "release": rel,
            "arch": arch,
            "summary": pkg.findtext(f"{{{RPM_NS}}}summary", ""),
            "size": size,
            "buildTime": build_time,
            "rpmFilename": filename,
            "changelog": [],
        }

    return packages


def parse_other(repodata_dir, packages):
    """Parse other.xml.gz and attach changelog entries to packages."""
    path = find_repodata_file(repodata_dir, "other")
    if not path:
        return

    with open_compressed(path) as f:
        tree = ET.parse(f)

    for pkg in tree.getroot().findall(f"{{{OTHER_NS}}}package"):
        name = pkg.get("name", "")
        version_el = pkg.find(f"{{{OTHER_NS}}}version")
        ver = version_el.get("ver", "") if version_el is not None else ""
        rel = version_el.get("rel", "") if version_el is not None else ""
        key = f"{name}-{ver}-{rel}"

        if key not in packages:
            continue

        for entry in pkg.findall(f"{{{OTHER_NS}}}changelog"):
            packages[key]["changelog"].append({
                "author": entry.get("author", ""),
                "date": int(entry.get("date", "0")),
                "text": (entry.text or "").strip(),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repodata-dir",
        required=True,
        help="path to the repodata/ directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="path to write packages.json",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="public base URL for the repo (e.g. https://rpm.lair.cafe/fedora/43/x86_64)",
    )
    args = parser.parse_args()

    packages = parse_primary(args.repodata_dir)
    parse_other(args.repodata_dir, packages)

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "baseUrl": args.base_url,
        "packages": list(packages.values()),
    }

    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {len(packages)} packages to {args.output}")


if __name__ == "__main__":
    main()
