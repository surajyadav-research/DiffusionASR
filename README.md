# Diffusion LLM-assisted Speech Recognition

The following workflow has been tested on an Ubuntu 20.04 LTS machine with 256 GB RAM and an NVIDIA H100 GPU (80GB VRAM).

# 0. install dependencies
1. create conda env and install libraries
    ```bash
    conda create -n salm python=3.11.11
    conda activate salm
    pip install -r requirements.txt
    ```
2. Add your HuggingFace token to the `./env.sh` file.

# 1. Download Datasets1
1. download the librispeech 100h training and all the validation and test sets from: https://www.openslr.org/12
2. create manifest files for all the sets using `./scripts/create_manifest_files.sh` similar to `./dummy-data/`.

# 2. Run Training
1. Change `./configs/librispeech/openai-whisper-large-v3_sln-large_llada-8b.yaml` to your desired config file.
2. Run training using the following command:
    ```bash
    ./scripts/train.sh ./configs/librispeech/openai-whisper-large-v3_sln-large_llada-8b.yaml
    ```

# 3. Run Inference
1. Run inference using the following command:
    ```bash
    ./scripts/infer.sh ./configs/librispeech/openai-whisper-large-v3_sln-large_llada-8b.yaml
    ```