"""
Model Module
    This module contains the SpeechEncoder2Adapter2Llm class, which is the main model class for the ASR task.
    Do not change the class defintion to add new speech encoders, llms or adapters.
    Instead, create a new class (and register) in the speech_encoder.py, llm.py and adapter.py.
"""

import os
from math import sqrt

import torch

from src.adapter import get_adapter_class
from src.llm import LLM, get_llm_class
from src.speech_encoder import SpeechEncoder, get_speech_encoder_class
from src.utils import compute_accuracy, get_model_size


class SpeechEncoder2Adapter2Llm(torch.nn.Module):
    def __init__(
        self,
        speech_encoder_name: str,
        llm_name: str,
        adapter_name: str,
        device="cuda:0",
        path_to_adapter_weights=None,
        lora_config_dict=None,
        lora_adapter_ckpt_path=None,
        **kwargs,
    ):
        super().__init__()
        self.device = device

        self.speech_encoder: SpeechEncoder = get_speech_encoder_class(
            speech_encoder_name
        )()
        self.llm: LLM = get_llm_class(llm_name)(device=device)
        self.tokenizer = self.llm.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # get dimensions from speech encoder to design the adapter
        self.speech_encoder_dim = self.speech_encoder.get_embedding_dim()
        self.llm_dim = self.llm.get_embedding_dim()

        # TODO: factory function abstraction is fuxy
        adapter_class = get_adapter_class(adapter_name)

        # just make sure if the speech encoder has get_num_blocks then adapter must have _add_auxilliary_input_conditioning_layers method
        if hasattr(self.speech_encoder, "get_num_blocks"):
            if not hasattr(adapter_class, "_add_auxilliary_input_conditioning_layers"):
                raise ValueError(
                    "Expected adapter to have _add_auxilliary_input_conditioning_layers method"
                )
            else:
                self.adapter = adapter_class(
                    speech_encoder_dim=self.speech_encoder_dim,
                    llm_dim=self.llm_dim,
                    path_to_adapter_weights=path_to_adapter_weights,
                    num_blocks=self.speech_encoder.get_num_blocks(),
                    num_mels=self.speech_encoder.get_num_mels(),
                )
        else:
            self.adapter = adapter_class(
                speech_encoder_dim=self.speech_encoder_dim,
                llm_dim=self.llm_dim,
                path_to_adapter_weights=path_to_adapter_weights,
            )

        print("number of trainable parameters in adapter", get_model_size(self.adapter))

        # Move components to device
        self.speech_encoder.to(device)
        self.adapter.to(device)

        # LoRA activation and loading
        if lora_config_dict is not None:
            self._init_lora_from_dict(lora_config_dict)
            if lora_adapter_ckpt_path is not None:
                print(f"Loading LoRA adapter weights from {lora_adapter_ckpt_path}")
                self.load_lora_adapter(lora_adapter_ckpt_path)
            print(
                "number of trainable parameters in LoRA adapter",
                sum(p.numel() for p in self.llm.model.parameters() if p.requires_grad),
            )

    def _init_lora_from_dict(self, lora_cfg):
        from peft import LoraConfig, TaskType

        task_type = getattr(TaskType, lora_cfg.get("task_type", "CAUSAL_LM"))
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=task_type,
        )
        self.activate_llm_lora(lora_config)

    def forward(self, batch, inference=False, use_matching_loss=False):
        input_ids = batch["input_ids"].to(self.device)
        prompt_lengths = batch["prompt_lengths"].to(self.device)
        # print("prompt_lengths", prompt_lengths)
        audio_starts = batch["audio_starts"].to(self.device)
        audio_ends = batch["audio_ends"].to(self.device)
        audio_mels = batch["audio_mels"].to(self.device)

        audio_mel_mask = torch.ones(
            audio_mels.shape[0], 3000, device=audio_mels.device, dtype=torch.bool
        )
        # [bs, max_num_samples] --> [bs, max_num_samples, speech_encoder_dim]
        speech_encoder_output = self.speech_encoder.encode_speech(
            audio_mels, audio_mel_mask
        )

        # [bs, max_num_samples, speech_encoder_dim] --> [bs, max_num_samples, adapter_dim]
        adapter_output = self.adapter.project(speech_encoder_output)

        # just so that model doesnt freak out at -1 token ids i.e. audio pseudo tokens
        input_ids[input_ids == -1] = self.tokenizer.pad_token_id

        # print("tokenizer.pad_token_id", self.tokenizer.pad_token_id)

        # [bs, max_num_tokens] --> [bs, max_num_tokens, llm_dim]
        input_embeds = self.llm.embed_tokens(input_ids)
        input_embeds = input_embeds.to(self.device)

        # slot the adapter output into the input embeds at positions where the audio pseudo tokens are
        for i in range(input_embeds.shape[0]):
            input_embeds[i, audio_starts[i] : audio_ends[i], :] = adapter_output[i]

        if inference:
            return input_embeds

        model_outputs = self.llm.train_step(
            inputs_embeds=input_embeds,
            prompt_lengths=prompt_lengths,
            input_ids=input_ids,
        )

        # calculate the token prediction accuracy based on masked tokens
        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            # Get masked indices from the forward process (stored in model_outputs)
            # Calculate accuracy based on masked tokens only
            masked_correct = 0
            masked_total = 0
            for i in range(input_ids.shape[0]):
                # Use response_gt_text and response_pred_text to compute accuracy
                gt_text = model_outputs.response_gt_text[i]
                pred_text = model_outputs.response_pred_text[i]
                # Token-level accuracy on masked positions
                gt_tokens = self.tokenizer.encode(gt_text, add_special_tokens=False)
                pred_tokens = self.tokenizer.encode(pred_text, add_special_tokens=False)
                min_len = min(len(gt_tokens), len(pred_tokens))
                if min_len > 0:
                    masked_correct += sum(
                        1
                        for g, p in zip(gt_tokens[:min_len], pred_tokens[:min_len])
                        if g == p
                    )
                    masked_total += len(gt_tokens)

            if masked_total > 0:
                acc = torch.tensor(masked_correct / masked_total, device=self.device)
            else:
                acc = torch.tensor(0.0, device=self.device)

        return (
            model_outputs,
            torch.tensor(0.0, device=self.device),
            acc,
        )  # matching loss is 0 for now

    def generate(self, batch): # 25.44 26.2 21.3 22.67
        input_embeds = self.forward(batch, inference=True)

        model_outputs = self.llm.generate(
            inputs_embeds=input_embeds,
        )
        return model_outputs

    def save_lora_adapter(self, path):
        from peft import get_peft_model_state_dict

        torch.save(get_peft_model_state_dict(self.llm.model), path)

    def load_lora_adapter(self, path):
        print(f"Loading LoRA adapter weights from {path}")
        from peft import set_peft_model_state_dict

        peft_model_state_dict = torch.load(path)
        set_peft_model_state_dict(self.llm.model, peft_model_state_dict)

    def activate_llm_lora(self, lora_config):
        self.llm.activate_lora(lora_config)
