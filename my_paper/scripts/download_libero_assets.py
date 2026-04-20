"""
Download LIBERO assets from HuggingFace Hub (with force re-download).

Usage:
    source /etc/network_turbo
    cd /root/autodl-tmp/robometer
    python my_paper/scripts/download_libero_assets.py
"""

import os
import shutil
import sys


def main():
    libero_pkg = os.path.dirname(
        os.path.abspath(
            __import__("libero.libero", fromlist=["__file__"]).__file__
        )
    )
    target_dir = os.path.join(libero_pkg, "assets")

    print(f"LIBERO package: {libero_pkg}")
    print(f"Target assets dir: {target_dir}")

    from huggingface_hub import snapshot_download

    repo_id = "jadechoghari/libero-assets"

    # Remove incomplete download to force fresh download
    incomplete_markers = [
        os.path.join(target_dir, "stable_scanned_objects", "plate", "plate.xml"),
        os.path.join(target_dir, "turbosquid_objects"),
    ]
    if not all(os.path.exists(m) for m in incomplete_markers):
        print("Previous download was incomplete. Force re-downloading...")
        force = True
    else:
        print("Assets look complete. Re-checking with HF Hub...")
        force = False

    print(f"\nDownloading from {repo_id} ...")
    print("This may take 5-10 minutes.\n")

    try:
        folder_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            force_download=force,
        )
    except Exception as e:
        print(f"\nError with default endpoint: {e}")
        print("\nRetrying with huggingface.co directly...")
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        folder_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            force_download=force,
        )

    print(f"\nDone! Assets at: {folder_path}")

    required_subdirs = [
        "articulated_objects", "stable_scanned_objects",
        "turbosquid_objects", "stable_hope_objects", "scenes",
    ]
    for d in required_subdirs:
        p = os.path.join(target_dir, d)
        if os.path.exists(p):
            n = sum(1 for _ in os.walk(p))
            print(f"  {d}: {n} directories")
        else:
            print(f"  {d}: MISSING!")

    # Verify critical file
    plate_xml = os.path.join(target_dir, "stable_scanned_objects", "plate", "plate.xml")
    print(f"\n  plate.xml exists: {os.path.exists(plate_xml)}")


if __name__ == "__main__":
    main()
