#!/usr/bin/env python3
"""
Collates SegmentProfiler JSON profiles from the 19Tow parametric sweep into a consolidated CSV and summary table.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path


KNOWN_STRAINS = {
    "green_lagrange",
    "linear",
    "engineering",
    "logarithmic",
}

KNOWN_CONTACTS = {
    "tanh",
    "quadratic",
    "piecewise_quadratic",
    "piecewise_linear",
    "exponential",
    "constant",
    "linear",
}


def is_19tow_file(fpath: Path, data: dict | None = None) -> bool:
    """
    Returns True if the file belongs to a 19Tow simulation and not other benchmarks (e.g. angle interlock).
    """
    fpath_str = str(fpath).lower()
    
    # Explicit exclusions
    exclusions = ["angle_interlock", "memory_profile", "threefiberspread", "singlefiber"]
    if any(ex in fpath_str for ex in exclusions):
        return False

    if data is not None:
        name = data.get("name", "")
        if name:
            if not ("19tow" in name.lower() or "rigid_mold" in name.lower()):
                return False
            return True

    if "rigid_mold" in fpath.parts or fpath.name.lower().startswith("19tow_") or re.match(r"NL\d+_", fpath.name, re.I):
        return True

    return False


def parse_filename_meta(filepath: str, data: dict | None = None):
    """
    Extracts metadata from profile paths such as:
      Nested: prof/rigid_mold/{date}/{strain}/{contact}/NL{nl}_{steps}steps_{solver}.json
      Flat:   prof/19tow_{contact}_{strain}_nl{nl}_{steps}steps.json
    """
    path = Path(filepath)
    parts = list(path.parts)
    stem = path.stem

    meta = {
        "date": "unknown",
        "strain": "unknown",
        "contact": "unknown",
        "nl": "",
        "steps": "",
        "solver": "CG",
    }

    # 1. Try parsing nested directory structure: .../<date>/<strain>/<contact>/<file>
    if len(parts) >= 4:
        potential_contact = parts[-2].lower()
        potential_strain = parts[-3].lower()
        potential_date = parts[-4]

        if potential_contact in KNOWN_CONTACTS:
            meta["contact"] = potential_contact
        if potential_strain in KNOWN_STRAINS:
            meta["strain"] = potential_strain
        meta["date"] = potential_date

    # 2. Try parsing stem like NL075_30steps_CG or NL125_60steps
    stem_match = re.search(r"NL(?P<nl>\d+)_(?P<steps>\d+)steps(?:_(?P<solver>[a-zA-Z0-9]+))?", stem, re.IGNORECASE)
    if stem_match:
        d = stem_match.groupdict()
        if d.get("nl"):
            meta["nl"] = str(int(d["nl"]))
        if d.get("steps"):
            meta["steps"] = str(int(d["steps"]))
        if d.get("solver"):
            meta["solver"] = d["solver"]

    # 3. Fallback to flat legacy pattern: 19tow_{contact}_{strain}_nl{nl}_{steps}steps
    flat_pattern = r"19tow_(?P<contact>.+)_(?P<strain>green_lagrange|linear|engineering|logarithmic)_nl(?P<nl>\d+)_(?P<steps>\d+)steps"
    flat_match = re.search(flat_pattern, stem, re.IGNORECASE)
    if flat_match:
        flat_dict = flat_match.groupdict()
        meta["contact"] = flat_dict["contact"]
        meta["strain"] = flat_dict["strain"]
        meta["nl"] = str(int(flat_dict["nl"]))
        meta["steps"] = str(int(flat_dict["steps"]))

    # 4. If fields still unknown, parse from data["name"] if available e.g. "19Tow (tanh, green_lagrange, NL=75, steps=30)"
    if data is not None and (meta["contact"] == "unknown" or meta["strain"] == "unknown"):
        prof_name = data.get("name", "")
        name_match = re.search(r"19Tow\s*\((?P<contact>[^,]+),\s*(?P<strain>[^,]+),\s*NL=(?P<nl>\d+),\s*steps=(?P<steps>\d+)\)", prof_name, re.I)
        if name_match:
            nd = name_match.groupdict()
            if meta["contact"] == "unknown":
                meta["contact"] = nd["contact"].strip().lower()
            if meta["strain"] == "unknown":
                meta["strain"] = nd["strain"].strip().lower()
            if not meta["nl"]:
                meta["nl"] = nd["nl"]
            if not meta["steps"]:
                meta["steps"] = nd["steps"]

    return meta


def collate_profiles(prof_dir: str = "prof", pattern: str = "**/*.json", output_csv: str = "prof/19tow_summary.csv"):
    p_dir = Path(prof_dir)
    if not p_dir.exists():
        print(f"Profile directory '{prof_dir}' does not exist.")
        return

    # Find candidate JSON files (ignoring non-profile and summary files)
    candidate_files = sorted([
        p for p in p_dir.glob(pattern)
        if p.is_file() and not p.name.endswith("_summary.json") and p.suffix == ".json"
    ])

    rows = []
    for fpath in candidate_files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception as e:
            continue

        # Filter strictly for 19Tow profiles
        if not is_19tow_file(fpath, data):
            continue

        meta = parse_filename_meta(str(fpath), data)
        try:
            summary = data.get("summary", {})
            nl_solve = summary.get("NONLINEAR_SOLVE", {})
            contact_search = summary.get("CONTACT_SEARCH", {})
            io_setup = summary.get("IO_SETUP", {})
            post_process = summary.get("POST_PROCESS", {})

            def _round(val):
                return round(val, 4) if isinstance(val, (int, float)) else None

            row = {
                "date": meta.get("date", ""),
                "contact_model": meta.get("contact", ""),
                "strain_measure": meta.get("strain", ""),
                "nl_max_iter": meta.get("nl", ""),
                "steps": meta.get("steps", ""),
                "solver": meta.get("solver", "CG"),
                "total_duration_s": _round(data.get("total_duration", 0.0)),
                "nl_solve_total_s": _round(nl_solve.get("total_time")),
                "nl_solve_avg_s": _round(nl_solve.get("avg_time")),
                "nl_solve_warm_avg_s": _round(nl_solve.get("warm_avg")),
                "nl_solve_cold_s": _round(nl_solve.get("cold_time")),
                "contact_search_total_s": _round(contact_search.get("total_time")),
                "contact_search_avg_s": _round(contact_search.get("avg_time")),
                "contact_search_warm_avg_s": _round(contact_search.get("warm_avg")),
                "contact_search_cold_s": _round(contact_search.get("cold_time")),
                "io_setup_total_s": _round(io_setup.get("total_time")),
                "post_process_total_s": _round(post_process.get("total_time")),
                "status": "COMPLETED",
                "file": str(fpath.relative_to(p_dir) if fpath.is_relative_to(p_dir) else fpath),
            }
        except Exception as e:
            row = {
                "date": meta.get("date", ""),
                "contact_model": meta.get("contact", ""),
                "strain_measure": meta.get("strain", ""),
                "nl_max_iter": meta.get("nl", ""),
                "steps": meta.get("steps", ""),
                "solver": meta.get("solver", "CG"),
                "total_duration_s": None,
                "nl_solve_total_s": None,
                "nl_solve_avg_s": None,
                "nl_solve_warm_avg_s": None,
                "nl_solve_cold_s": None,
                "contact_search_total_s": None,
                "contact_search_avg_s": None,
                "contact_search_warm_avg_s": None,
                "contact_search_cold_s": None,
                "io_setup_total_s": None,
                "post_process_total_s": None,
                "status": f"ERROR: {e}",
                "file": str(fpath.relative_to(p_dir) if fpath.is_relative_to(p_dir) else fpath),
            }
        rows.append(row)

    if not rows:
        print(f"No 19Tow profile JSON files found matching '{pattern}' in '{prof_dir}'.")
        return

    fieldnames = [
        "date",
        "contact_model",
        "strain_measure",
        "nl_max_iter",
        "steps",
        "solver",
        "total_duration_s",
        "nl_solve_avg_s",
        "nl_solve_warm_avg_s",
        "nl_solve_total_s",
        "contact_search_avg_s",
        "contact_search_warm_avg_s",
        "contact_search_total_s",
        "io_setup_total_s",
        "post_process_total_s",
        "status",
        "file",
    ]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCollation complete. Processed {len(rows)} 19Tow profiles -> {output_csv}")
    print("\n" + "=" * 120)
    print(
        f"{'Date':<8} | {'Contact Model':<20} | {'Strain Measure':<16} | "
        f"{'Total (s)':<10} | {'Avg NL (s)':<11} | {'Avg Contact (s)':<16} | {'Status'}"
    )
    print("-" * 120)
    for r in rows:
        tot = f"{r['total_duration_s']:.2f}" if r['total_duration_s'] is not None else "N/A"
        nl_avg = f"{r['nl_solve_avg_s']:.3f}" if r['nl_solve_avg_s'] is not None else "N/A"
        contact_avg = f"{r['contact_search_avg_s']:.4f}" if r['contact_search_avg_s'] is not None else "N/A"
        print(
            f"{r['date']:<8} | {r['contact_model']:<20} | {r['strain_measure']:<16} | "
            f"{tot:<10} | {nl_avg:<11} | {contact_avg:<16} | {r['status']}"
        )
    print("=" * 120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate 19Tow SegmentProfiler JSONs into a summary CSV.")
    parser.add_argument("--prof-dir", default="prof", help="Directory containing profile JSONs (default: prof)")
    parser.add_argument("--pattern", default="**/*.json", help="Glob pattern for profile files (default: **/*.json)")
    parser.add_argument("--output-csv", default="prof/19tow_summary.csv", help="Output summary CSV (default: prof/19tow_summary.csv)")
    args = parser.parse_args()

    collate_profiles(args.prof_dir, args.pattern, args.output_csv)
