# Wav2Vec2 Sparse Autoencoder Analysis

The goal of this project is to train a sparse autoencoder on the activations of a wav2vec2 model and then interpret the latent space.

## Usage Pipeline

This codebase follows these main steps:

1. **Data Collection**: Get data from LibriSpeech to run the wav2vec2 model
   - See [`process_data.py`](./process_data.py)

2. **Activation Collection**: Run the wav2vec2 model on the data and collect ~20M activations
   - See [`collect_activations.py`](./collect_activations.py)

3. **Model Training**: Train a sparse autoencoder on the activations
   - Model architecture: [`model.py`](./model.py)
   - Training logic: [`trainer.py`](./trainer.py)
   - Training script: [`train.py`](./train.py)

4. **Data Processing**: Convert the original dataset into 2s segments for easier visualization.
   - See [`split_audio_dataset.py`](./split_audio_dataset.py)

5. **Feature Interpretation**: Analyze the latent space by examining top activations for each SAE feature
   - See [`interpret_features.ipynb`](./interpret_features.ipynb)

## Resources

### Interactive Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LrsE9-GoEplb5PbWf-gSIbjq8vPzObO4?usp=sharing)

### Pretrained Models and Data
- [Trained SAE](https://huggingface.co/pavanyellow/wave-sae/blob/main/sae-resid-layer-6.pt)
- [Split dataset](https://huggingface.co/datasets/pavanyellow/librispeech_asr)

---

**Note**: This codebase is made public only for reference. Best way to use it is to copy the relevant parts and modify them for your own use case.
