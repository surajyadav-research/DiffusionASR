# iterate over all the test gt and preds files
# calculate the metrics
# save the metrics to a file

import os
import json
from src.utils import calculate_wer_score, calculate_rtfx_score


PATH_TO_TEST_GT_PREDS = "speech-align-llm/benchmarks/checkpoints/whisper-large-v3"

def calculate_metrics(gt_file, pred_file):
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    # calculate the metrics
    metrics = {}
    metrics['wer'] = calculate_wer_score(gt_data, pred_data)
    # metrics['cer'] = calculate_cer_score(gt_data, pred_data)
    return metrics

def main():
    # iterate over all the test gt and preds files
    for gt_file in os.listdir('test_gt'):
        for pred_file in os.listdir('test_preds'):
            metrics = calculate_metrics(gt_file, pred_file)
            print(metrics)

if __name__ == '__main__':
    main()