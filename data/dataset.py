import os
import json
import tarfile
from pathlib import Path
import torchaudio
import io
from huggingface_hub import snapshot_download
import subprocess
import numpy as np
import soundfile as sf


'''
request for gpu on psc

interact -p GPU-shared --gres=gpu:v100-32:1 -t 06:00:00 -A cis250145p
interact -p RM-shared --ntasks-per-node=1 --cpus-per-task=32 -t 8:00:00 -A cis250145p
jupyter notebook --no-browser --ip=0.0.0.0
'''

os.environ['HF_HOME'] = '/ocean/projects/cis250145p/tanghang/ASR_adapt/hf_cache'

def download_hf_dataset(repo_id, local_dir):
    print("Downloading dataset...")
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type='dataset', local_dir=local_dir)
    print(f"✅ Dataset downloaded to {local_dir}")

class SimpleTarredASRDataset:
    def __init__(self, data_dir, split='train', target_sr=16000, max_samples=5):
        self.split = split
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.max_samples = max_samples

        self.tarred_dir = self.data_dir / f"{split}_tarred" / "sharded_manifests_with_image"
        self.audio_shards_dir = self.tarred_dir / "audio_shards"
        self.samples = self._load_manifests()
        self.shard_to_tar = self._map_shards()

    def _load_manifests(self):
        manifest_files = list(self.tarred_dir.glob("manifest_*.json"))
        samples = []
        for mf in manifest_files:
            with open(mf, 'r', encoding='utf-8') as f:
                for line in f:
                    manifest = json.loads(line.strip())
                    if 'audio_filepath' not in manifest or 'shard_id' not in manifest:
                        continue
                    if self.split == 'test':
                        samples.append(manifest)
                    else:
                        text_value = manifest.get('text')
                        if text_value is not None and str(text_value).strip():
                            samples.append(manifest)
                    if self.max_samples and len(samples) >= self.max_samples:
                        break
            if self.max_samples and len(samples) >= self.max_samples:
                break
        return samples

    def _map_shards(self):
        shard_to_tar = {}
        for tar_path in self.audio_shards_dir.glob("audio_*.tar.xz"):
            try:
                shard_id = int(Path(tar_path.stem).stem.split('_')[1])
                shard_to_tar[shard_id] = tar_path
            except Exception:
                continue
        return shard_to_tar

    def __getitem__(self, idx):
        manifest = self.samples[idx]
        shard_id = manifest['shard_id']
        audio_file = manifest['audio_filepath']
        tar_path = self.shard_to_tar.get(shard_id)
        if not tar_path:
            return None
        try:
            with tarfile.open(tar_path, 'r:xz') as tar:
                member = tar.getmember(audio_file)
                audio_bytes = tar.extractfile(member).read()
                # Decode webm bytes to PCM via ffmpeg binary (subprocess) for robustness
                proc = subprocess.run(
                    [
                        'ffmpeg', '-threads', '1', '-i', 'pipe:0', '-f', 'wav', '-acodec', 'pcm_s16le',
                        '-ac', '1', '-ar', str(self.target_sr), 'pipe:1'
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
                text = manifest.get('text', '') or ''
                return {
                    'audio': waveform,
                    'text': text,
                    'duration': manifest.get('duration', 0.0),
                    'audio_filepath': audio_file,
                    'shard_id': shard_id,
                    'gender': manifest.get('gender', ''),
                    'age_group': manifest.get('age_group', ''),
                    'location': manifest.get('location', '')
                }
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

class WavDirASRDataset:
    """
    Dataset for loading pre-converted wav files from a directory.
    Expects a manifest file with 'audio_filepath', 'text', and other metadata fields.
    All audio files should be mono and 16kHz.
    """
    def __init__(self, manifest_path, audio_dir=None, max_samples=None):
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                manifest = json.loads(line.strip())
                if 'audio_filepath' not in manifest:
                    continue
                if audio_dir:
                    manifest['audio_filepath'] = str(Path(audio_dir) / Path(manifest['audio_filepath']).name)
                self.samples.append(manifest)
                if max_samples and len(self.samples) >= max_samples:
                    break

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample['audio_filepath']
        try:
            waveform, sr = sf.read(audio_path, dtype='float32')
            if waveform.ndim > 1:
                waveform = waveform[0]
            text = sample.get('text', '') or ''
            return {
                'audio': waveform,
                'text': text,
                'duration': sample.get('duration', 0.0),
                'audio_filepath': audio_path,
                'shard_id': sample.get('shard_id', None),
                'gender': sample.get('gender', ''),
                'age_group': sample.get('age_group', ''),
                'location': sample.get('location', '')
            }
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    repo_id = "DigitalUmuganda/ASR_Fellowship_Challenge_Dataset"
    local_dir = "/ocean/projects/cis250145p/tanghang/ASR_adapt/ASR_adapter/metadata/"
    # download_hf_dataset(repo_id, local_dir)
    for split in ['train', 'val', 'test']:
        print(f"\nVerifying {split} split:")
        dataset = SimpleTarredASRDataset(local_dir, split=split, max_samples=1)
        print(f"Loaded {len(dataset)} samples")
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample:
                text_preview = sample['text'][:50] if sample['text'] else '[NO TEXT]'
                print(f"Sample {i}: text={text_preview}... audio_shape={sample['audio'].shape} file={sample['audio_filepath']}")