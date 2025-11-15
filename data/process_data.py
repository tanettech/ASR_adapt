import os
import json
import tarfile
from pathlib import Path
import io
from datasets import Dataset, concatenate_datasets
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import subprocess
import soundfile as sf
import numpy as np
import gc
import psutil

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1

#interact -p RM-shared --ntasks-per-node=32 -t 06:00:00



def process_manifest_entry(manifest, tar, target_sr=16000, split='train'):
    """Process a single manifest given an already-open tarfile.TarFile object.
    Returns a sample dict or None on error.
    """
    audio_file = manifest['audio_filepath']
    try:
        member = tar.getmember(audio_file)
        audio_bytes = tar.extractfile(member).read()
        # Decode webm to mono 16kHz WAV using ffmpeg binary (subprocess). Limit ffmpeg threads to 1.
        proc = subprocess.run(
            [
                'ffmpeg', '-threads', '1', '-i', 'pipe:0', '-f', 'wav', '-acodec', 'pcm_s16le',
                '-ac', '1', '-ar', str(target_sr), 'pipe:1'
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {proc.stderr.decode(errors='ignore')}")
        waveform, sr = sf.read(io.BytesIO(proc.stdout), dtype='float32')
        waveform = np.array(waveform)
        if waveform.ndim > 1:
            waveform = waveform[0]

        # Handle text: for test split, text can be null
        text = manifest.get('text', '') or ''

        return {
            'audio': waveform,
            'text': text,
            'duration': manifest.get('duration', len(waveform) / target_sr),
            'audio_filepath': audio_file,
            'shard_id': manifest.get('shard_id'),
            'gender': manifest.get('gender', ''),
            'age_group': manifest.get('age_group', ''),
            'location': manifest.get('location', '')
        }
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None


def process_shard_worker(sid, tar_path, data_dir, split='train', target_sr=16000):
    """Top-level worker to process a shard. Reads manifest files from disk
    for the given `split` and processes entries whose `shard_id` == sid.
    This avoids pickling large manifest lists into the process pool."""
    results = []
    try:
        tar_path = str(tar_path)
        data_dir = Path(data_dir)
        tarred_dir = data_dir / f"{split}_tarred" / "sharded_manifests_with_image"
        # Count expected manifests for logging
        nman = 0
        for mf in sorted(tarred_dir.glob("manifest_*.json")):
            with open(mf, 'r', encoding='utf-8') as f:
                for line in f:
                    ls = line.strip()
                    if not ls:
                        continue
                    try:
                        jm = json.loads(ls)
                    except Exception:
                        continue
                    if 'shard_id' in jm and int(jm['shard_id']) == int(sid):
                        nman += 1

        print(f"Shard {sid} starting: {nman} manifests; tar={tar_path}", flush=True)
        seen = 0
        with tarfile.open(tar_path, 'r:xz') as tar:
            for mf in sorted(tarred_dir.glob("manifest_*.json")):
                with open(mf, 'r', encoding='utf-8') as f:
                    for line in f:
                        line_stripped = line.strip()
                        if not line_stripped:
                            continue
                        try:
                            manifest = json.loads(line_stripped)
                        except Exception:
                            continue
                        if 'shard_id' not in manifest:
                            continue
                        if int(manifest['shard_id']) != int(sid):
                            continue
                        # For test split, accept even if text is null
                        if split != 'test':
                            text_value = manifest.get('text')
                            if text_value is None or not str(text_value).strip():
                                seen += 1
                                if seen % 100 == 0:
                                    print(f"Shard {sid}: processed {seen}/{nman}", flush=True)
                                continue

                        r = process_manifest_entry(manifest, tar, target_sr=target_sr, split=split)
                        if r:
                            results.append(r)
                        seen += 1
                        if seen % 100 == 0:
                            print(f"Shard {sid}: processed {seen}/{nman}", flush=True)

        print(f"Shard {sid} finished: produced {len(results)} samples", flush=True)
    except Exception as e:
        print(f"Error processing shard {sid} ({tar_path}): {e}")
    
    # Force garbage collection to free memory
    gc.collect()
    return results

def process_tarred_to_hf(
    data_dir: Path,
    split: str = 'train',
    target_sr: int = 16000,
    out_dir: Path = None,
    num_proc: int = 2,
    batch_size: int = 1000
):
    data_dir = Path(data_dir)
    tarred_dir = data_dir / f"{split}_tarred" / "sharded_manifests_with_image"
    audio_shards_dir = tarred_dir / "audio_shards"
    manifest_files = sorted(tarred_dir.glob("manifest_*.json"))
    # Map shard_id to tar file
    shard_to_tar = {}
    for tar_path in audio_shards_dir.glob("audio_*.tar.xz"):
        try:
            shard_id = int(Path(tar_path.stem).stem.split('_')[1])
            shard_to_tar[shard_id] = tar_path
        except Exception:
            continue

    # Count total manifests (we'll have workers read per-shard manifests from disk to avoid
    # pickling large lists into the process pool). This reduces memory and serialization overhead.
    total = 0
    for mf in manifest_files:
        with open(mf, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                manifest = json.loads(line_stripped)
                if 'audio_filepath' not in manifest or 'shard_id' not in manifest:
                    continue
                if split == 'test':
                    pass_ok = True
                else:
                    text_value = manifest.get('text')
                    pass_ok = text_value is not None and str(text_value).strip()
                if not pass_ok:
                    continue
                total += 1

    shard_ids = sorted(shard_to_tar.keys())
    print(f"Total manifests for {split}: {total} across {len(shard_ids)} shards")
    
    # Log initial memory usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Initial memory usage: {mem_info.rss / 1024**3:.2f} GB")

    out_dir = Path(out_dir) if out_dir else (data_dir / f"{split}_hf_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_dir_base = out_dir / "batches"
    batch_dir_base.mkdir(parents=True, exist_ok=True)

    batch = []
    batch_count = 0
    processed = 0

    # shard worker moved to module-level `process_shard_worker` to be picklable

    # Submit one task per shard (parallel across shards). Workers will read
    # their shard manifests from disk to avoid pickling large lists. If the
    # process pool fails (e.g. workers killed by the system), fall back to
    # sequential processing so the job can continue.
    try:
        with ProcessPoolExecutor(max_workers=num_proc) as executor:
            futures = {}
            for sid in shard_ids:
                tar_path = shard_to_tar.get(sid)
                if not tar_path:
                    print(f"Warning: no tar file for shard {sid}")
                    continue
                futures[executor.submit(process_shard_worker, sid, str(tar_path), str(data_dir), split, target_sr)] = sid

            # Use a tqdm progress bar in the main process to track samples processed.
            with tqdm(total=total, desc=f"Processing {split}", unit='samples') as pbar:
                for future in as_completed(futures):
                    sid = futures[future]
                    try:
                        results = future.result()
                    except Exception as e:
                        print(f"Shard {sid} failed: {e}")
                        results = []
                    
                    # Log memory after each shard completes
                    mem_info = process.memory_info()
                    print(f"Memory after shard {sid}: {mem_info.rss / 1024**3:.2f} GB")
                    gc.collect()

                    nnew = 0
                    for r in results:
                        batch.append(r)
                        processed += 1
                        nnew += 1

                        if len(batch) >= batch_size:
                            # flush batch to disk
                            batch_count += 1
                            batch_out = batch_dir_base / f"batch_{batch_count}"
                            Dataset.from_list(batch).save_to_disk(str(batch_out))
                            print(f"Wrote batch {batch_count} ({len(batch)} samples) -> {batch_out}")
                            batch = []

                    # update progress bar by number of samples returned by this shard
                    if nnew:
                        pbar.update(nnew)
    except Exception as exc:
        print(f"Parallel processing failed: {exc}. Falling back to sequential processing.")
        # Sequential fallback with tqdm
        with tqdm(total=total, desc=f"Processing {split} (sequential)", unit='samples') as pbar:
            for sid in shard_ids:
                tar_path = shard_to_tar.get(sid)
                if not tar_path:
                    print(f"Warning: no tar file for shard {sid}")
                    continue
                try:
                    results = process_shard_worker(sid, str(tar_path), str(data_dir), split, target_sr=target_sr)
                except Exception as e:
                    print(f"Shard {sid} failed in sequential mode: {e}")
                    results = []

                for r in results:
                    batch.append(r)
                    processed += 1
                    pbar.update(1)

                    if len(batch) >= batch_size:
                        batch_count += 1
                        batch_out = batch_dir_base / f"batch_{batch_count}"
                        Dataset.from_list(batch).save_to_disk(str(batch_out))
                        print(f"Wrote batch {batch_count} ({len(batch)} samples) -> {batch_out}")
                        batch = []

    # flush remaining
    if batch:
        batch_count += 1
        batch_out = batch_dir_base / f"batch_{batch_count}"
        Dataset.from_list(batch).save_to_disk(str(batch_out))
        print(f"Wrote final batch {batch_count} ({len(batch)} samples) -> {batch_out}")

    # Concatenate batches into final dataset (load each and concat)
    batch_paths = sorted(batch_dir_base.glob('batch_*'))
    if not batch_paths:
        print("No batches written, nothing to save.")
        return

    ds_list = []
    for p in batch_paths:
        try:
            ds_list.append(Dataset.load_from_disk(str(p)))
        except Exception as e:
            print(f"Failed to load batch {p}: {e}")

    if not ds_list:
        print("No batches could be loaded; aborting final save.")
        return

    if len(ds_list) == 1:
        final_ds = ds_list[0]
    else:
        try:
            final_ds = concatenate_datasets(ds_list)
        except Exception as e:
            print(f"Concatenation failed: {e}. Saving batches individually.")
            print(f"Final processed data is available in {batch_dir_base}")
            return

    final_out = out_dir / 'final'
    final_ds.save_to_disk(str(final_out))
    print(f"✅ Processed dataset saved to {final_out}")

if __name__ == "__main__":
    data_dir = "/ocean/projects/cis250145p/tanghang/ASR_adapt/ASR_adapter/metadata/"
    
    # Process train split only (val and test already completed)
    print(f"\n{'='*60}")
    print(f"Processing train split...")
    print(f"{'='*60}")
    process_tarred_to_hf(data_dir, split='train', num_proc=2, batch_size=1000)