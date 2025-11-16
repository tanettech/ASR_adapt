import os
from datasets import load_from_disk
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
# Play the audio using librosa
import librosa
import librosa.display
import soundfile as sf
# Define dataset paths
splits = {
    "train": "/ocean/projects/cis250145p/tanghang/ASR_adapt/ASR_adapter/metadata/train_hf_dataset/final",
    "val": "/ocean/projects/cis250145p/tanghang/ASR_adapt/ASR_adapter/metadata/val_hf_dataset/final",
    "test": "/ocean/projects/cis250145p/tanghang/ASR_adapt/ASR_adapter/metadata/test_hf_dataset/final",
}

def verify_audio_text_pairing(dataset, split_name, n_samples=1):
    """
    Verify audio-text pairing for a dataset split.

    Args:
        dataset: The dataset split to verify.
        split_name: Name of the split (train, val, test).
        n_samples: Number of samples to verify.
    """
    print(f"\n🔍 Verifying {split_name.upper()} split...")
    print(f"Total samples: {len(dataset)}")
    print("=" * 70)

    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i + 1}/{n_samples}")
        print("-" * 70)

        # Display text (if available)
        if split_name != "test":
            print(f"Text: {sample['text']}")

        # Load audio
        if isinstance(sample["audio"], list):
            audio_data = np.array(sample["audio"], dtype=np.float32)
            sample_rate = 16000  # Assuming a default sample rate
        elif isinstance(sample["audio"], dict) and "array" in sample["audio"]:
            audio_data = np.array(sample["audio"]["array"], dtype=np.float32)
            sample_rate = sample["audio"].get("sampling_rate", 16000)
        else:
            print("❌ ERROR: Audio field does not contain expected data structure.")
            continue

        print(f"Audio data shape: {audio_data.shape}")
        print(f"Duration: {sample['duration']} seconds")
        plt.figure(figsize=(10, 4))
        plt.plot(np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data)), audio_data)
        plt.title(f"Waveform for Sample {i + 1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()


        import librosa
        import librosa.display
        import soundfile as sf

        print("Playing audio using librosa...")
        sf.write("temp_audio.wav", audio_data, sample_rate)  # Save audio to a temporary file
        os.system("ffplay -nodisp -autoexit temp_audio.wav")  # Play audio using ffplay
        #delay for me to play the audio
        input("Press Enter to continue...")
        os.remove("temp_audio.wav")  # Clean up the temporary file

        # Plot the waveform
 
# Load datasets
print("\nStep 1: Loading datasets...")
train_dataset = load_from_disk(splits["train"])
val_dataset = load_from_disk(splits["val"])
test_dataset = load_from_disk(splits["test"])
print("✅ Datasets loaded successfully!")

# Verify each split
verify_audio_text_pairing(train_dataset, "train", n_samples=1)
verify_audio_text_pairing(val_dataset, "val", n_samples=1)
verify_audio_text_pairing(test_dataset, "test", n_samples=1)