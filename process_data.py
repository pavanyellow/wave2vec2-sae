import os
import glob
from datasets import Dataset, concatenate_datasets  
import soundfile as sf
import torchaudio


def load_librispeech(base_path="audio_data/LibriSpeech", split="dev-clean"):
    """Load LibriSpeech dataset from local directory.
    
    Args:
        base_path: Root directory containing LibriSpeech data
        split: Dataset split to load (e.g., 'dev-clean', 'test-clean', 'train-clean-100')
    
    Returns:
        datasets.Dataset object
    """
    examples = {}
    transcripts_glob = os.path.join(base_path, split, "*/*/*.txt")
    batch = 0
    for transcript_file in glob.glob(transcripts_glob):
        path = os.path.dirname(transcript_file)
        
        with open(transcript_file, "r") as f:
            for line in f:
                line = line.strip()
                key, transcript = line.split(" ", 1)
                audio_file = f"{key}.flac"
                speaker_id, chapter_id = [int(el) for el in key.split("-")[:2]]
                
                example = {
                    "id": key,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "file": os.path.join(path, audio_file),
                    "text": transcript,
                }
                
                # Load audio data
                audio_path = os.path.join(path, audio_file)
                if os.path.exists(audio_path):
                    audio_data, sampling_rate = sf.read(audio_path)
                    example["audio"] = {
                        "array": audio_data,
                        "sampling_rate": sampling_rate
                    }
                
                examples[key] = example

                if len(examples) > 3000:
                    save_path = f"audio_data/{split}.hf"
                    temp_path = f"audio_data/{split}_temp.hf"
                    
                    if os.path.exists(save_path) and batch > 0:
                        existing_dataset = Dataset.load_from_disk(save_path)
                        new_dataset = Dataset.from_dict({
                            k: [v[k] for v in examples.values()]
                            for k in examples[next(iter(examples))].keys()
                        })
                        dataset = concatenate_datasets([existing_dataset, new_dataset])
                    else:
                        dataset = Dataset.from_dict({
                            k: [v[k] for v in examples.values()]
                            for k in examples[next(iter(examples))].keys()
                        })
                    
                    print(f"Writing batch {batch} to disk")
                    dataset.save_to_disk(temp_path)
                    
                    if os.path.exists(save_path):
                        import shutil
                        shutil.rmtree(save_path)
                    
                    os.rename(temp_path, save_path)
                    
                    batch += 1
                    examples = {}

    dataset = Dataset.from_dict({
        k: [v[k] for v in examples.values()]
        for k in examples[next(iter(examples))].keys()
    })
    
    return dataset

# Load the dataset
dataset = load_librispeech(split="train-clean-100")
# dataset.save_to_disk("audio_data/train-clean-100.hf")
# dataset = Dataset.load_from_disk("audio_data/train-clean-100.hf")

