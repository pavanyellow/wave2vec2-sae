import os
import glob
from datasets import Dataset
import soundfile as sf

def load_librispeech(base_path="audio_data/LibriSpeech", split="dev-clean", segment_length=2.0):
    examples = {}
    transcripts_glob = os.path.join(base_path, split, "*/*/*.txt")
    batch = 0
    example_id = 0

    files_processed = 0
    
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
                
                audio_path = os.path.join(path, audio_file)
                if os.path.exists(audio_path):
                    files_processed += 1
                    if files_processed % 100 == 0:
                        print(f"Processed {files_processed} files")
                    if len(examples) > 15000:
                        break

                    audio_data, sampling_rate = sf.read(audio_path)
                    sampling_rate = 16000
                    
                    samples_per_segment = int(segment_length * sampling_rate)
                    total_samples = len(audio_data)
                    
                    for start_idx in range(0, total_samples, samples_per_segment):
                        end_idx = min(start_idx + samples_per_segment, total_samples)
                        
                        if end_idx - start_idx < samples_per_segment:
                            continue
                            
                        segment = audio_data[start_idx:end_idx]
                        
                        segment_key = f"{key}_{start_idx//samples_per_segment}"
                        
                        example = {
                            "id": segment_key,
                            "speaker_id": speaker_id,
                            "chapter_id": chapter_id,
                            "file": os.path.join(path, audio_file),
                            "text": transcript,
                            "audio": {
                                "array": segment,
                                "sampling_rate": sampling_rate
                            },
                            "start_sample": start_idx,
                            "end_sample": end_idx
                        }
                        
                        examples[segment_key] = example
                        example_id += 1
                        break

    dataset = Dataset.from_dict({
        k: [v[k] for v in examples.values()]
        for k in examples[next(iter(examples))].keys()
    })
    
    save_path = f"audio_data/{split}-2s.hf"
    dataset.save_to_disk(save_path)
    
    return dataset

if __name__ == "__main__":
    dataset = load_librispeech(base_path="audio_data/LibriSpeech", split="train-clean-100", segment_length=2.0)
    print(dataset)