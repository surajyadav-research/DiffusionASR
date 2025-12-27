"""
Code to benchmark the performance of whisper-large-v3 model on the indicsuperb dataset

"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import yaml
import os
import warnings
from tqdm import tqdm
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import calculate_wer_score
from dataset import BenchmarkDataset

# Suppress the specific FutureWarning about deprecated 'inputs' parameter
warnings.filterwarnings("ignore", category=FutureWarning, message=".*The input name `inputs` is deprecated.*")

device = "cuda:0"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_ID = "openai/whisper-large-v3"
PATH_TO_CONFIG = "configs/benchmarks/whisper_large.yaml"

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# load config and create dataset and dataloader
config = load_config(PATH_TO_CONFIG)
model_name = config["model_name"]

# load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, local_files_only=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)

# Create pipeline for ASR
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=False,  # Set to True if you want timestamps
    generate_kwargs={"language": "hindi", "task": "transcribe"},
)

def get_transcription(audio_raw):
    result = pipe(audio_raw)
    predicted_text = result["text"]
    return predicted_text

# run inference
for path_to_test_data in config["paths"]["list_of_test_manifest_files"]:
    test_data_name = os.path.splitext(os.path.basename(path_to_test_data))[0]
    path_to_save_gt_file = os.path.join("benchmarks", "checkpoints", model_name, f"{test_data_name}_gt")
    path_to_save_pred_file = os.path.join(
        "benchmarks", "checkpoints", model_name, f"{test_data_name}_pred"
    )

    path_to_test_manifest_file = os.path.join(config["paths"]["data_dir"], path_to_test_data)
    print(f"Processing: {path_to_test_manifest_file}")
    
    dataset = BenchmarkDataset(path_to_test_manifest_file)
    
    key2pred = {}
    key2gt = {}

    for index_, (audio_raw, target, key) in tqdm(enumerate(dataset), total=len(dataset)):

        # Run inference
        predicted_text = get_transcription(audio_raw)
        
        key2pred[key] = predicted_text
        key2gt[key] = target

    print(f"Predictions: {len(key2pred)}")
    print(f"Ground truths: {len(key2gt)}")

    os.makedirs(os.path.dirname(path_to_save_pred_file), exist_ok=True)
    with open(path_to_save_pred_file, "w", encoding="utf-8") as f:
        for key, pred in key2pred.items():
            pred = pred.strip().replace("|", "")
            f.write(f"{key}\t{pred}\n")
    
    # overwrite if exists
    os.makedirs(os.path.dirname(path_to_save_gt_file), exist_ok=True)
    with open(path_to_save_gt_file, "w", encoding="utf-8") as f:
        for key, gt in key2gt.items():
            gt = gt.strip().replace("|", "")
            f.write(f"{key}\t{gt}\n")

    # calculate the metrics
    wer_value = 0
    for key, gt in key2gt.items():
        pred = key2pred[key]
        wer = calculate_wer_score(gt, pred)
        wer_value += wer
    
    wer_value = wer_value / len(key2gt)
    print(f"WER: {wer_value}")

    if not os.path.exists(os.path.join(os.path.dirname(path_to_save_pred_file), "best_model_no_loss_all_wer_results.json")):
        data = {}
    else:   
        with open(os.path.join(os.path.dirname(path_to_save_pred_file), "best_model_no_loss_all_wer_results.json"), "r") as f:
            data = json.load(f)
            data[test_data_name] = wer_value
    with open(os.path.join(os.path.dirname(path_to_save_pred_file), "best_model_no_loss_all_wer_results.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    
    print(f"Saved predictions to: {path_to_save_pred_file}")
    print(f"Saved ground truths to: {path_to_save_gt_file}") 