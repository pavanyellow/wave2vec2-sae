from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import Dataset
import torch
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)


#data_split = "dev-clean"
data_split = "train-clean-100"
datadir = f"audio_data/{data_split}.hf"
dataset = Dataset.load_from_disk(datadir)
print(f"Loaded {len(dataset)} datapoints from {datadir}")

class ActivationCapturer:
    def __init__(self):
        self.activations_buffer = torch.tensor([]).to(device)
        self.index = 6
    
    def clear_buffer(self):
        del self.activations_buffer
        self.activations_buffer = torch.tensor([]).to(device)
        torch.cuda.empty_cache()
        
    def hook_fn(self, module, input, output):
        resid = output[0]
        #print(resid.shape)
        self.activations_buffer = torch.cat((self.activations_buffer, resid[0]), dim=0)
        if self.activations_buffer.shape[0] > 2000000:
            self.activations_buffer = self.activations_buffer[torch.randperm(self.activations_buffer.shape[0])]
            print(f"Saving buffer of size {self.activations_buffer.shape}")
            torch.save(self.activations_buffer, f"activations/{data_split}-{self.index}.pt")
            self.index += 1
            self.clear_buffer()


capturer = ActivationCapturer()
layer_idx = 5

hook = model.wav2vec2.encoder.layers[layer_idx].register_forward_hook(capturer.hook_fn)
total_datapoints = len(dataset)

for i in tqdm( range(17000,total_datapoints)):
    input_values = processor(dataset[i]["audio"]["array"], return_tensors="pt", padding="longest", sampling_rate=16000).input_values.to(device)  # Batch size 1

    with torch.no_grad():
        _ = model(input_values)

hook.remove()

print(f"Found {capturer.activations_buffer.shape[0]} activations from {total_datapoints} datapoints")
capturer.activations_buffer = capturer.activations_buffer[torch.randperm(capturer.activations_buffer.shape[0])]
save_dir = f"activations/{data_split}-{capturer.index}.pt"

torch.save(capturer.activations_buffer, save_dir)




        


