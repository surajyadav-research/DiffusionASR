import torch
from transformers import AutoModelForCTC
from pyctcdecode import build_ctcdecoder
import json
from dataset import BenchmarkDataset
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import calculate_wer_score
import yaml

PATH_TO_CONFIG = "configs/benchmarks/indicwav2vec-hindi.yaml"

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def processor(audio_numpy):
    audio = torch.from_numpy(audio_numpy) # shape is [16000 * n]
    audio = torch.nn.functional.layer_norm(audio, audio.shape) # shape is [n]
    audio = audio.unsqueeze(0).to("cuda:0") # shape is [1, n]
    return audio

with open("pretrained-speech-encoders/indicwav2vec-hindi/vocab.json", "r") as f:
    vocab_dict = json.load(f)
vocab_list = [token for token, idx in sorted(vocab_dict.items(), key=lambda x: x[1])]
decoder = build_ctcdecoder(vocab_list)

model = AutoModelForCTC.from_pretrained("pretrained-speech-encoders/indicwav2vec-hindi", trust_remote_code=True, local_files_only=True)
model.to("cuda:0")
model.eval()

config = load_config(PATH_TO_CONFIG)
model_name = config["model_name"]


def get_transcription(audio_raw):
    audio = processor(audio_raw)
    with torch.no_grad():
        logits = model(audio).logits.cpu().squeeze(0).numpy()
    output = decoder.decode(logits, beam_width=100)
    return output

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
        data[test_data_name] = wer_value
    else:   
        with open(os.path.join(os.path.dirname(path_to_save_pred_file), "best_model_no_loss_all_wer_results.json"), "r") as f:
            data = json.load(f)
            data[test_data_name] = wer_value
    with open(os.path.join(os.path.dirname(path_to_save_pred_file), "best_model_no_loss_all_wer_results.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    
    print(f"Saved predictions to: {path_to_save_pred_file}")
    print(f"Saved ground truths to: {path_to_save_gt_file}") 