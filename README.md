# F5: microbudget training

**F5-TTS**: Diffusion Transformer with ConvNeXt V2, faster trained and inference.


**Sway Sampling**: Inference-time flow step sampling strategy, greatly improves performance


### Create a separate environment if needed

```bash
# Create a conda env with python_version>=3.10  (you could also use virtualenv)
conda create -n f5-tts python=3.10 --no-default-packages -y
conda activate f5-tts
pip install uv
uv pip install -r requirements.txt
```

### Install PyTorch with matched device

<details>
<summary>NVIDIA GPU</summary>

> ```bash
> # Install pytorch with your CUDA version, e.g.
> pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
> ```

</details>


### Runtime

Deployment solution with Triton and TensorRT-LLM.

#### Benchmark Results
Decoding on a single L20 GPU, using 26 different prompt_audio & target_text pairs, 16 NFE.

| Model               | Concurrency    | Avg Latency | RTF    | Mode            |
|---------------------|----------------|-------------|--------|-----------------|
| F5-TTS Base (Vocos) | 2              | 253 ms      | 0.0394 | Client-Server   |
| F5-TTS Base (Vocos) | 1 (Batch_size) | -           | 0.0402 | Offline TRT-LLM |
| F5-TTS Base (Vocos) | 1 (Batch_size) | -           | 0.1467 | Offline Pytorch |

See [detailed instructions](src/f5_tts/runtime/triton_trtllm/README.md) for more information.


## Inference

- In order to achieve desired performance, take a moment to read [detailed guidance](src/f5_tts/infer).

## Training

### 1. With Hugging Face Accelerate

Refer to [training & finetuning guidance](src/f5_tts/train) for best practice.


Read [training & finetuning guidance](src/f5_tts/train) for more instructions.


## [Evaluation](src/f5_tts/eval)
