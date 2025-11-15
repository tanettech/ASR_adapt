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

#interact -p RM-shared --ntasks-per-node=64 -t 08:00:00

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


def process_shard_worker(sid, tar_path, data_dir, split='train', target_sr=16000, batch_dir=None, chunk_size=500):
    """Process a shard and save in chunks to minimize memory usage.
    Returns the list of chunk paths saved."""
    chunk_paths = []
    chunk = []
    chunk_idx = 0
    
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
                            chunk.append(r)
                        
                        seen += 1
                        if seen % 100 == 0:
                            print(f"Shard {sid}: processed {seen}/{nman}", flush=True)
                        
                        # Save chunk when it reaches chunk_size
                        if len(chunk) >= chunk_size and batch_dir:
                            chunk_path = Path(batch_dir) / f"shard_{sid}_chunk_{chunk_idx}"
                            Dataset.from_list(chunk).save_to_disk(str(chunk_path))
                            chunk_paths.append(chunk_path)
                            print(f"  → Saved chunk {chunk_idx} ({len(chunk)} samples) to {chunk_path}", flush=True)
                            chunk = []
                            chunk_idx += 1
                            gc.collect()

        # Save remaining samples in chunk
        if chunk and batch_dir:
            chunk_path = Path(batch_dir) / f"shard_{sid}_chunk_{chunk_idx}"
            Dataset.from_list(chunk).save_to_disk(str(chunk_path))
            chunk_paths.append(chunk_path)
            print(f"  → Saved final chunk {chunk_idx} ({len(chunk)} samples) to {chunk_path}", flush=True)
            chunk = []
            gc.collect()

        print(f"Shard {sid} finished: saved {len(chunk_paths)} chunks", flush=True)
        
    except Exception as e:
        print(f"Error processing shard {sid} ({tar_path}): {e}")
    
    # Force garbage collection to free memory
    gc.collect()
    return chunk_paths

def process_tarred_to_hf(
    data_dir: Path,
    split: str = 'train',
    target_sr: int = 16000,
    out_dir: Path = None,
    num_proc: int = 1,
    batch_size: int = 500
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

    # Sequential processing: process each shard in chunks to minimize memory usage
    print("Processing shards sequentially with chunked saving to minimize memory usage...")
    all_chunk_paths = []
    
    for sid in shard_ids:
        tar_path = shard_to_tar.get(sid)
        if not tar_path:
            print(f"Warning: no tar file for shard {sid}")
            continue
        
        try:
            # Process shard with chunked saving (saves every 500 samples)
            chunk_paths = process_shard_worker(
                sid, 
                str(tar_path), 
                str(data_dir), 
                split, 
                target_sr=target_sr,
                batch_dir=str(batch_dir_base),
                chunk_size=500
            )
            
            if chunk_paths:
                all_chunk_paths.extend(chunk_paths)
                print(f"✓ Shard {sid} completed: {len(chunk_paths)} chunks saved")
            else:
                print(f"✗ Shard {sid} produced no chunks")
            
            # Log memory after each shard
            mem_info = process.memory_info()
            print(f"Memory after shard {sid}: {mem_info.rss / 1024**3:.2f} GB\n")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"✗ Shard {sid} failed: {e}\n")

    # Concatenate all chunks into final dataset
    print(f"\n{'='*60}")
    print(f"Concatenating {len(all_chunk_paths)} chunks into final dataset...")
    print(f"{'='*60}")
    
    if not all_chunk_paths:
        print("No chunks written, nothing to save.")
        return

    ds_list = []
    for i, p in enumerate(all_chunk_paths):
        try:
            ds_list.append(Dataset.load_from_disk(str(p)))
            if (i + 1) % 10 == 0:
                print(f"Loaded {i + 1}/{len(all_chunk_paths)} chunks...")
        except Exception as e:
            print(f"Failed to load chunk {p}: {e}")

    if not ds_list:
        print("No chunks could be loaded; aborting final save.")
        return

    print(f"Concatenating {len(ds_list)} datasets...")
    if len(ds_list) == 1:
        final_ds = ds_list[0]
    else:
        try:
            final_ds = concatenate_datasets(ds_list)
        except Exception as e:
            print(f"Concatenation failed: {e}. Chunks are available in {batch_dir_base}")
            return

    final_out = out_dir / 'final'
    print(f"Saving final dataset to {final_out}...")
    final_ds.save_to_disk(str(final_out))
    print(f"✅ Processed dataset saved to {final_out}")
    print(f"✅ Total samples: {len(final_ds)}")

if __name__ == "__main__":
    data_dir = "/ocean/projects/cis250145p/tanghang/ASR_adapt/ASR_adapter/metadata/"
    
    # Process train split only (val and test already completed)
    print(f"\n{'='*60}")
    print(f"Processing train split...")
    print(f"{'='*60}")
    process_tarred_to_hf(data_dir, split='train', num_proc=1, batch_size=1000)