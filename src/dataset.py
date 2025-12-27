import copy
import json

import numpy as np
import torch

from whisper import whisper


class AsrDataset4LLaDA(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer=None,
        prompt=None,
        mel_size=80,
        fix_length_audio=-1,
        fix_audio_duration=-1,
        inference_mode=False,
        normalize=False,
        input_type="mel",
        target_column="target",
        path_to_jsonl_file=None,
        llm_name=None,
    ):
        """
        Initialize the speech dataset.

        Args:
            dataset_config: Configuration dictionary for the dataset
            tokenizer: Tokenizer to convert text to/from tokens
            split: Dataset split to use ('train' or other, typically 'val')
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.llm_name = llm_name
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = prompt
        self.mel_size = mel_size
        self.fix_length_audio = fix_length_audio
        self.fix_audio_duration = fix_audio_duration
        self.inference_mode = inference_mode
        self.normalize = normalize
        self.input_type = input_type
        self.target_column = target_column
        self.tokenizer.eot_token_id = 126348

        self.data_list = []
        with open(path_to_jsonl_file, encoding="utf-8") as fin:
            for line in fin:
                try:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)
                except:
                    continue

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data_list)

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        """
        data_dict = self.data_list[index]
        audio_path = data_dict.get("source")
        target = data_dict.get(self.target_column, None).lower()
        key = data_dict.get("key", None)

        # load audio and convert to mel
        audio_raw = whisper.load_audio(audio_path)
        audio_duration = len(audio_raw) / 16000  # Duration in seconds before padding
        audio_raw = whisper.pad_or_trim(audio_raw, 30 * 16000)
        audio_mel = whisper.log_mel_spectrogram(
            audio_raw, n_mels=self.mel_size
        ).permute(1, 0)
        audio_length = (audio_mel.shape[0] + 1) // 2  # Whisper downsample factor
        audio_length = audio_length // 5  # Additional downsampling

        # create audio placeholder
        audio_pseudo = torch.full(
            (audio_length,), -1
        )  # will always be 300 becuase of whisper's 30s input
        assert audio_length == 300, "Audio length should be 300"

        # create prompt
        prompt = """Transcribe the following speech to text:  """.lower()
        message = [{"role": "user", "content": prompt}]
        prompt_ids = self.tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True
        )
        eot_id = self.tokenizer.eot_token_id
        eot_idx = prompt_ids.index(eot_id)
        prompt_ids = prompt_ids[:eot_idx] + audio_pseudo.tolist() + prompt_ids[eot_idx:]
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
        prompt_length = prompt_ids.shape[0]

        # get the prompt offset (num tokens before the audio pseudo tokens)
        audio_start = eot_idx
        audio_end = eot_idx + len(audio_pseudo)
        # create labels
        target_ids = (
            self.tokenizer.encode(target, add_special_tokens=False)
            + [self.tokenizer.eot_token_id]
            + [self.tokenizer.eos_token_id]
        )
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        # prompt ids + audio pseudo tokens + target ids
        full_ids = torch.cat((prompt_ids, target_ids))

        # print the decoded full_ids
        # sample_ids = full_ids.clone()
        # sample_ids[sample_ids == -1] = 0
        # print("Decoded full_ids:", self.tokenizer.decode(sample_ids.tolist()))

        return (
            full_ids,
            prompt_length,
            audio_start,
            audio_end,
            audio_mel,
            audio_duration,
            key,
            target,
        )

    def collator(self, samples):
        max_len = max(ids.shape[0] for ids, _, _, _, _, _, _, _ in samples)
        padded_input_ids = torch.zeros(len(samples), max_len, dtype=torch.long)
        prompt_lengths = torch.zeros(len(samples), dtype=torch.long)
        audio_starts = torch.zeros(len(samples), dtype=torch.long)
        audio_ends = torch.zeros(len(samples), dtype=torch.long)
        audio_mels = []
        audio_durations = []
        keys = []
        targets = []
        for i, (
            ids,
            prompt_len,
            audio_start,
            audio_end,
            audio_mel_tensor,
            audio_duration,
            key,
            target,
        ) in enumerate(samples):
            padded_input_ids[i, : ids.shape[0]] = ids
            padded_input_ids[i, ids.shape[0] :] = self.tokenizer.pad_token_id
            prompt_lengths[i] = prompt_len
            audio_starts[i] = audio_start
            audio_ends[i] = audio_end
            audio_mels.append(audio_mel_tensor)
            audio_durations.append(audio_duration)
            keys.append(key)
            targets.append(target)
        audio_mels = torch.stack(audio_mels).permute(0, 2, 1)
        audio_durations = torch.tensor(audio_durations)
        return {
            "input_ids": padded_input_ids,
            "prompt_lengths": prompt_lengths,
            "audio_starts": audio_starts,
            "audio_ends": audio_ends,
            "audio_mels": audio_mels,
            "audio_durations": audio_durations,
            "keys": keys,
            "targets": targets,
        }
