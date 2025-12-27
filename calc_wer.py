import os

from src.utils import read_and_calculate_wer

# Path to the directory containing prediction and gt files
adapter_dir = "/mnt/disk-n8/checkpoints/whisper-ft-encoder_full-decoder_full__hi"
adapter_base = "best_model_no_loss"

# Define the test subsets and their suffixes
subsets = [
    ("hindi_noisy_test", "hindi_noisy_test"),
    ("hindi_noisy_test_known", "hindi_noisy_test_known"),
    ("hindi_test", "hindi_test"),
    ("hindi_test_known", "hindi_test_known"),
]

wer_scores = {}
for subset_name, subset_suffix in subsets:
    pred_file = os.path.join(adapter_dir, f"{adapter_base}_{subset_suffix}_pred")
    gt_file = os.path.join(adapter_dir, f"{adapter_base}_{subset_suffix}_gt")
    # Reuse subset_name as the logical name for reporting
    results = read_and_calculate_wer(gt_file, pred_file, test_data_name=subset_name)
    wer_scores.update(results)
    print(f"{subset_name}: WER = {results.get(subset_name)}")

print(f"\nAll subset WERs for {adapter_base} in {adapter_dir}:")
for subset, score in wer_scores.items():
    print(f"{subset}: {score}")
