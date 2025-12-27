"""
Usage:
    python create_manifest_files.py --kaldi_dir KALDI_DIR --output_audio_dir OUTPUT_DIR

Also create another manifest file with the following format:
{"key": "hindi-[audio_id]_ASR", "source": audio_path, "target": transcription}
{"key": "hindi-844424931194893-977-m_ASR", "source": "/home/puneet/speech-align-llm/data/kb_data_clean_wav/hindi/test_known/audio/977/844424931194893-977-m.wav", "target": "इस अवसर पर गोरेगांव की अनेक महिलाओं ने अपनी उपस्थिति दर्ज कराई"}
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

from lhotse import CutSet, load_kaldi_data_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create manifest files for speech data from Kaldi format"
    )
    parser.add_argument(
        "--kaldi_dir",
        type=str,
        required=True,
        help="Path to Kaldi-style data directory (containing text, segments, wav.scp, etc.)",
    )
    parser.add_argument(
        "--output_audio_dir",
        type=str,
        required=True,
        help="Path to save split .wav audio files",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16000,
        help="Expected sampling rate of the audio (default: 16000)",
    )
    return parser.parse_args()


def create_manifest_from_kaldi(
    kaldi_dir: str,
    output_audio_dir: str,
    output_manifest_path: str,
    sampling_rate: int = 16_000,
) -> Tuple[int, Path, Path]:
    """
    Converts Kaldi-style dataset into split audio segments and a JSONL manifest.

    Args:
        kaldi_dir: Path to Kaldi-style data dir (containing text, segments, wav.scp, etc.)
        output_audio_dir: Path to save split .wav audio files.
        output_manifest_path: Path to save the JSONL manifest.
        sampling_rate: Expected sampling rate of the audio (default 16kHz).

    Returns:
        Tuple containing:
        - Number of audio segments created
        - Path to output audio directory
        - Path to output manifest file
    """
    manifests = load_kaldi_data_dir(kaldi_dir, sampling_rate=sampling_rate)

    raw_cuts = CutSet.from_manifests(
        recordings=manifests[0], supervisions=manifests[1], lazy=False
    )
    cuts = raw_cuts.trim_to_supervisions(num_jobs=4)

    output_audio_dir = Path(output_audio_dir)
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    cuts.save_audios(storage_path=output_audio_dir, format="wav")

    output_manifest_path = Path(output_manifest_path)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        for cut in cuts:
            entry = {
                "audio_path": str(output_audio_dir / f"{cut.id}.wav"),
                "text": cut.supervisions[0].text,
                "speaker": cut.supervisions[0].speaker,
                "duration": cut.duration,
                "cut_id": cut.id,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"Saved {len(cuts)} audio segments to {output_audio_dir}\nJSONL manifest saved to {output_manifest_path}"
    )

    return len(cuts), output_audio_dir, output_manifest_path


def main():
    args = parse_args()
    output_manifest_path = Path(args.output_audio_dir) / "manifest.jsonl"

    create_manifest_from_kaldi(
        kaldi_dir=args.kaldi_dir,
        output_audio_dir=args.output_audio_dir,
        output_manifest_path=str(output_manifest_path),
        sampling_rate=args.sampling_rate,
    )

    # read the manifest file and create another manifest file with the following format:
    # {"key": "hindi-[audio_id]_ASR", "source": audio_path, "target": transcription}
    # {"key": "hindi-844424931194893-977-m_ASR", "source": "/home/puneet/speech-align-llm/data/kb_data_clean_wav/hindi/test_known/audio/977/84444931194893-977-m.wav", "target": "इस अवसर पर गोरेगांव की अनेक महिलाओं ने अपनी उपस्थिति दर्ज कराई"}
    # original format: {"audio_path": "/home/puneet/speech-align-llm/mucs/data/processed/blindtest/848297_1Ex3WEqCQ7VGfqIn_0004.wav", "text": "आपको movie clip का audio सुनने के लिए headset और speakers की ज़रुरत होगी", "speaker": "848297", "duration": 5.0, "cut_id": "848297_1Ex3WEqCQ7VGfqIn_0004"}
    new_manifest_path = Path(args.output_audio_dir) / "manifest_ASR.jsonl"
    lines = []
    with open(output_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            # NOTE: data cleanup rules:
            # 1. assert audio durations must be [0.5, 30.0] seconds
            # 2. assert text must not be empty
            if (
                entry["text"] == ""
                or entry["duration"] < 0.5
                or entry["duration"] > 30.0
            ):
                continue
            # need to correct the audio path becuase lhotse fucked up the paths in manifest jsonl files despite creating it
            # Extract the cut_id and create correct path structure
            cut_id = entry["cut_id"]
            speaker_id = cut_id.split("_")[
                0
            ]  # e.g. "654444" from "654444_0049FmqkWQMNW6Tc_0000"
            speaker_prefix = speaker_id[:3]  # First 3 digits
            entry["audio_path"] = os.path.join(
                str(args.output_audio_dir), speaker_prefix, f"{cut_id}.wav"
            )
            lines.append(
                f'{{"key": "hindi-{entry["cut_id"]}_ASR", "source": "{entry["audio_path"]}", "target": "{entry["text"]}"}}'
            )
    with open(new_manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
