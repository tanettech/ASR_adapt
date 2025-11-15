#!/usr/bin/env python3
"""Extract a single audio member from a shard tar, convert to mono 16kHz WAV, and optionally SCP it to a remote host.

Usage examples:
  # extract and convert one file to /tmp/sample.wav
  python extract_and_copy.py --data-dir /path/to/metadata --split train --shard-id 0 \
    --audio-file audio_1742015136-C7IF7b5C3kSEwT3kzDO3n1cQ5S13.webm --out /tmp/sample.wav
  # extract and scp to local machine
  python extract_and_copy.py --data-dir /path/to/metadata --split train --shard-id 0 \
    --audio-file audio_1742015136-C7IF7b5C3kSEwT3kzDO3n1cQ5S13.webm --out /tmp/sample.wav \
    --scp user@laptop:/home/user/Downloads/

This script requires the `ffmpeg` binary to be available on PATH.
"""

import argparse
import tarfile
from pathlib import Path
import subprocess
import sys
import shutil


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg binary not found on PATH. Install it or load module (e.g. 'module load ffmpeg').")
        return False
    return True


def find_member_in_tar(tar, name):
    # try exact name first
    try:
        return tar.getmember(name)
    except KeyError:
        # try to find a member whose name ends with the provided name
        for m in tar.getmembers():
            if m.name.endswith(name):
                return m
    return None


def find_member_across_shards(data_dir, split, audio_filename):
    """Search all shard tar.xz files for a member whose name ends with audio_filename.
    Returns tuple (tar_path, member) or (None, None) if not found.
    """
    data_dir = Path(data_dir)
    tarred_dir = data_dir / f"{split}_tarred" / "sharded_manifests_with_image" / "audio_shards"
    if not tarred_dir.exists():
        return None, None

    for tar_path in sorted(tarred_dir.glob('audio_*.tar.xz')):
        try:
            with tarfile.open(tar_path, 'r:xz') as tar:
                for m in tar.getmembers():
                    if m.name.endswith(audio_filename):
                        return tar_path, m
        except Exception:
            # skip unreadable/Corrupt tar files silently
            continue
    return None, None


def extract_and_convert(data_dir, split, shard_id, audio_filename, out_path, target_sr=16000):
    data_dir = Path(data_dir)
    tarred_dir = data_dir / f"{split}_tarred" / "sharded_manifests_with_image" / "audio_shards"
    tar_name = f"audio_{shard_id}.tar.xz"
    tar_path = tarred_dir / tar_name

    if not tar_path.exists():
        raise FileNotFoundError(f"Shard tar not found: {tar_path}")

    print(f"Opening shard: {tar_path}")
    with tarfile.open(tar_path, 'r:xz') as tar:
        member = find_member_in_tar(tar, audio_filename)
        if member is None:
            print(f"Member '{audio_filename}' not found in {tar_path}.")
            # try to locate the member across other shard tarballs
            print("Searching other shards for the requested member...")
            found_tar, found_member = find_member_across_shards(data_dir, split, audio_filename)
            if found_tar is not None:
                print(f"Found member in: {found_tar} -> {found_member.name}")
                # extract from the found tar instead
                with tarfile.open(found_tar, 'r:xz') as tar2:
                    fobj = tar2.extractfile(found_member)
                    if fobj is None:
                        raise RuntimeError(f"Failed to extract member {found_member.name} from {found_tar}")
                    audio_bytes = fobj.read()
            else:
                print("Available sample members (first 20 in the originally requested shard):")
                for m in tar.getmembers()[:20]:
                    print("  ", m.name)
                raise KeyError(f"Audio member not found: {audio_filename}")

        print(f"Extracting member: {member.name}")
        fobj = tar.extractfile(member)
        if fobj is None:
            raise RuntimeError(f"Failed to extract member {member.name} from {tar_path}")
        else:
            audio_bytes = fobj.read()

    # run ffmpeg and write stdout to out_path
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg', '-y', '-i', 'pipe:0',
        '-f', 'wav', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(target_sr), 'pipe:1'
    ]

    print(f"Running ffmpeg to produce: {out_path}")
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        stdout, stderr = proc.communicate(input=audio_bytes)
        if proc.returncode != 0:
            print("ffmpeg failed (stderr):")
            print(stderr.decode(errors='ignore'))
            raise RuntimeError("ffmpeg conversion failed")
        # write wav bytes to file
        out_path.write_bytes(stdout)

    print(f"WAV written to: {out_path}")
    return out_path


def scp_file(local_path, remote_target):
    print(f"Copying {local_path} -> {remote_target}")
    cmd = ['scp', str(local_path), remote_target]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"scp failed with code {r.returncode}")
    print("scp completed")


def main():
    p = argparse.ArgumentParser(description="Extract and convert single audio member from a shard tar.")
    p.add_argument('--data-dir', required=True, help='Path to metadata directory (contains train_tarred, val_tarred, test_tarred)')
    p.add_argument('--split', default='train', choices=['train','val','test','validation'], help='Split name')
    p.add_argument('--shard-id', required=True, type=int, help='Shard id (integer), e.g. 0')
    p.add_argument('--audio-file', required=True, help='Audio filename from manifest, e.g. audio_1743051032-...webm')
    p.add_argument('--out', default='/tmp/sample.wav', help='Output WAV path')
    p.add_argument('--scp', default=None, help='Optional scp target (user@host:/path/)')
    p.add_argument('--sr', default=16000, type=int, help='Target sample rate')

    args = p.parse_args()

    if args.split == 'validation':
        args.split = 'val'

    if not check_ffmpeg():
        sys.exit(1)

    try:
        out_path = extract_and_convert(args.data_dir, args.split, args.shard_id, args.audio_file, args.out, target_sr=args.sr)
    except Exception as e:
        print('Error:', e)
        sys.exit(2)

    if args.scp:
        try:
            scp_file(out_path, args.scp)
        except Exception as e:
            print('scp failed:', e)
            sys.exit(3)


if __name__ == '__main__':
    main()
