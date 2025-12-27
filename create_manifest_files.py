"""
Usage:
    python create_manifest_files.py
Arguments:
    --config CONFIG_PATH    Path to the config file (default: config.yaml)
    --output-dir OUTPUT_DIR Override the output directory from config
"""

import argparse
import json
import os
import re
import string
from pathlib import Path

import yaml
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create manifest files for speech data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file (default: config.yaml)",
    )
    parser.add_argument(
        "--path_to_output_manifests_dir",
        type=str,
        default=None,
        help="Override the output directory from config",
    )
    return parser.parse_args()


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# NOQA: https://github.com/AI4Bharat/IndicSUPERB/blob/8860e5c39c726186a0a36f2163cf30397f398e0d/utilities/preprocess_asr.py
def process_transcription(raw_text, language):
    """
    Process a raw transcription string with all the operations from the original code.

    Args:
        raw_text (str): The raw transcription text
        language (str): Language name (e.g., 'hindi', 'bengali', etc.)

    Returns:
        str: Processed transcription text
    """

    # Language code mapping
    lang2lcode = {
        "bengali": "bn",
        "gujarati": "gu",
        "hindi": "hi",
        "kannada": "kn",
        "malayalam": "ml",
        "marathi": "mr",
        "odia": "or",
        "punjabi": "pa",
        "sanskrit": "sa",
        "tamil": "ta",
        "telugu": "te",
        "urdu": "ur",
    }

    # Step 1: Strip whitespace and specific punctuation marks
    text = raw_text.strip().strip("॥").strip("৷").strip("।").strip()

    # Step 2: Get language code
    lang_code = lang2lcode.get(language.lower(), "hi")  # default to hindi if not found

    # Step 3: Remove all punctuation including Devanagari danda (।)
    text = text.translate(str.maketrans("", "", string.punctuation + "।"))

    # Step 4: Normalize digit spacing - add spaces around digits
    text = re.sub(r" ?(\d+) ?", r" \1 ", text)

    # Step 5: Apply Indic normalization (except for Urdu)
    if lang_code not in ["ur"]:
        try:

            factory = IndicNormalizerFactory()
            normalizer = factory.get_normalizer(lang_code)
            text = normalizer.normalize(text)
        except ImportError:
            print("Warning: indicnlp not available, skipping Indic normalization")
            pass

    return text


def main():
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    data_dir = config["paths"]["data_dir"]

    # Define paths using config
    path_to_clean_dir = os.path.join(data_dir, "kb_data_clean_wav")
    path_to_noisy_dir = os.path.join(data_dir, "kb_data_noisy_wav")

    # Use output directory from args if provided, otherwise from config
    path_to_output_manifests_dir = (
        args.path_to_output_manifests_dir
        if args.path_to_output_manifests_dir
        else os.path.join(data_dir, "indicsuperb-manifests")
    )

    # Create output directory if it doesn't exist
    os.makedirs(path_to_output_manifests_dir, exist_ok=True)

    # Process clean data
    for language in Path(path_to_clean_dir).iterdir():
        if language.is_dir():
            for split in language.iterdir():
                manifest = []
                if split.is_dir():
                    audio_id2text = {}
                    transcription_path = split / "transcription_n2w.txt"
                    with open(transcription_path, "r") as f:
                        for line in f:
                            audio_id, text = line.strip().split("\t")
                            audio_id2text[audio_id] = text

                    path_to_speaker_dir = Path(split) / "audio"
                    if not path_to_speaker_dir.exists():
                        continue

                    for speaker in path_to_speaker_dir.iterdir():
                        if speaker.is_dir():
                            for audio_path in speaker.iterdir():
                                filename = str(audio_path).split("/")[-1]
                                filename_mfa = filename.split(".")[0] + ".m4a"
                                transcription = audio_id2text[filename_mfa]
                                language_name = str(language).split("/")[-1]
                                filename_key = (
                                    language_name
                                    + "-"
                                    + filename.split(".")[0]
                                    + "_ASR"
                                )
                                audio_path = str(audio_path)

                                entry = {
                                    "key": filename_key,
                                    "source": audio_path,
                                    "target": transcription,
                                    "processed_target": process_transcription(
                                        transcription, language_name
                                    ),
                                }
                                manifest.append(entry)

                split_name = str(split).split("/")[-1]
                path_to_language_split_manifest_file = os.path.join(
                    path_to_output_manifests_dir, f"{language_name}_{split_name}.jsonl"
                )
                with open(path_to_language_split_manifest_file, "w") as f:
                    for entry in manifest:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"Manifest saved at {path_to_language_split_manifest_file}")

    # Process noisy data
    for language in Path(path_to_noisy_dir).iterdir():
        if language.is_dir():
            for split in language.iterdir():
                manifest = []
                if split.is_dir():
                    audio_id2text = {}
                    transcription_path = split / "transcription_n2w.txt"
                    with open(transcription_path, "r") as f:
                        for line in f:
                            audio_id, text = line.strip().split("\t")
                            audio_id2text[audio_id] = text

                    path_to_speaker_dir = Path(split) / "audio"
                    if not path_to_speaker_dir.exists():
                        continue

                    for speaker in path_to_speaker_dir.iterdir():
                        if speaker.is_dir():
                            for audio_path in speaker.iterdir():
                                filename = str(audio_path).split("/")[-1]
                                filename_mfa = filename.split(".")[0] + ".m4a"
                                transcription = audio_id2text[filename_mfa]
                                language_name = str(language).split("/")[-1]
                                filename_key = (
                                    language_name
                                    + "-"
                                    + filename.split(".")[0]
                                    + "_ASR"
                                )
                                audio_path = str(audio_path)

                                entry = {
                                    "key": filename_key,
                                    "source": audio_path,
                                    "target": transcription,
                                    "processed_target": process_transcription(
                                        transcription, language_name
                                    ),
                                }
                                manifest.append(entry)

                split_name = "noisy_" + str(split).split("/")[-1]
                path_to_language_split_manifest_file = os.path.join(
                    path_to_output_manifests_dir, f"{language_name}_{split_name}.jsonl"
                )
                with open(path_to_language_split_manifest_file, "w") as f:
                    for entry in manifest:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"Manifest saved at {path_to_language_split_manifest_file}")


if __name__ == "__main__":
    main()
