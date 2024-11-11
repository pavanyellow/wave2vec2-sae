from datasets import Dataset

dataset = Dataset.load_from_disk("audio_data/train-clean-100-2s.hf")
print(dataset[0])
print(len(dataset))