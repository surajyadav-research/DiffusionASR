"""
torch dataset shared by all the benchmarks
manifest format: {"key": "hindi-844424931226803-164-f_ASR", "source": "/mnt/disk-b6/data/INDICSUPERB/kb_data_noisy_wav/hindi/test_known/audio/164/844424931226803-164-f.wav", "target": "वैसे भी शिकायतकर्ता बिहारी और एसओ दोनों बागपत के ही हैं", "processed_target": "वैसे भी शिकायतकर्ता बिहारी और एसओ दोनों बागपत के ही हैं"}

"""

import torch
from transformers import AutoModelForCTC, AutoProcessor
import torchaudio.functional as F
import json
from whisper import whisper

class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_jsonl_file):
        self.data_list = []
        with open(path_to_jsonl_file, encoding="utf-8") as fin:
            for line in fin:
                try:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)
                except:
                    continue

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_dict = self.data_list[index]
        audio_path = data_dict["source"]
        audio_raw = whisper.load_audio(audio_path, sr=16000)
        return audio_raw, data_dict["processed_target"], data_dict["key"]


