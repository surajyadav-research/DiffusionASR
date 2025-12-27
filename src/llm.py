"""
LLM Module

To Extend:
    1. Make a new class subclassing `LLM` with the following methods:
        - `__init__`: Initialize the LLM
        - `get_embedding_dim`: Get the dimension of the embedding
        - `embed_tokens`: Embed the tokens
        - [optional] `generate`: Generate the text
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Keep PEFT for LoRA only
try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA will not work.")

from src.prefix_tuning import SimplePrefixTuning

LLM_REGISTRY = {}


def register_llm(name):
    """
    Decorator to register LLM classes in the LLM_REGISTRY dictionary.

    Args:
        name (str): The name of the LLM.
    """

    def wrapper(cls):
        LLM_REGISTRY[name] = cls
        return cls

    return wrapper


def get_llm_class(name):
    """Get LLM class by name from registry"""
    if name not in LLM_REGISTRY:
        available_llms = list(LLM_REGISTRY.keys())
        raise ValueError(
            f"Unknown LLM class: {name}. Available options: {available_llms}"
        )
    return LLM_REGISTRY[name]


class LLM(ABC, torch.nn.Module):
    def __init__(
        self,
        path_to_pretrained_model=None,
        freeze_layers=True,
        device="cuda:0",
    ):
        super().__init__()
        if path_to_pretrained_model is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                path_to_pretrained_model,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                path_to_pretrained_model,
                use_fast=True,
                trust_remote_code=True,
                local_files_only=True,
            )
            if freeze_layers:
                for _, param in self.model.named_parameters():
                    param.requires_grad = False
            self.model.eval()
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.is_lora_active = False
        self.is_prefix_tuning_active = False
        self.prefix_tuning = None

    def activate_lora(self, lora_config=None):
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is required for LoRA but not available.")
        if lora_config is None:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        self.model = get_peft_model(self.model, lora_config)
        self.is_lora_active = True
        print("LoRA adapters activated.")

    def activate_prefix_tuning(self, prefix_tuning_config=None):
        """
        Activate custom prefix tuning (embedding-based approach)

        Args:
            prefix_tuning_config: Dict with keys:
                - num_virtual_tokens (int): Number of virtual tokens (default: 20)
                - prefix_projection (bool): Use MLP reparameterization (default: False)
                - prefix_projection_hidden_size (int): Hidden size for MLP (default: 512)
                - prefix_dropout (float): Dropout rate (default: 0.0)
        """
        if prefix_tuning_config is None:
            prefix_tuning_config = {
                "num_virtual_tokens": 20,
                "prefix_projection": False,
                "prefix_projection_hidden_size": 512,
                "prefix_dropout": 0.0,
            }

        # Get model hidden size
        hidden_size = self.model.config.hidden_size

        # Create prefix tuning module (embedding-based, simpler than KV-cache approach)
        self.prefix_tuning = SimplePrefixTuning(
            num_virtual_tokens=prefix_tuning_config.get("num_virtual_tokens", 20),
            hidden_size=hidden_size,
            prefix_projection=prefix_tuning_config.get("prefix_projection", False),
            prefix_projection_hidden_size=prefix_tuning_config.get(
                "prefix_projection_hidden_size", 512
            ),
            prefix_dropout=prefix_tuning_config.get("prefix_dropout", 0.0),
        )

        # Move to same device as model
        self.prefix_tuning = self.prefix_tuning.to(self.model.device)

        # Only prefix tuning parameters should be trainable
        for param in self.prefix_tuning.parameters():
            param.requires_grad = True

        self.is_prefix_tuning_active = True
        self.num_virtual_tokens = prefix_tuning_config.get("num_virtual_tokens", 20)

        print(
            f"Prefix-tuning activated with {self.num_virtual_tokens} virtual tokens (embedding-based)."
        )
        print(
            f"Prefix projection: {prefix_tuning_config.get('prefix_projection', False)}"
        )

    def forward(self, inputs_embeds, attention_mask, labels):
        # NOTE: simply override the forward method for the diffusion llm
        if self.is_prefix_tuning_active:
            # Get batch size
            batch_size = inputs_embeds.shape[0]

            # Get prefix embeddings: [batch_size, num_virtual_tokens, hidden_size]
            prefix_embeds = self.prefix_tuning(batch_size)
            # print(f"inputs_embeds shape: {inputs_embeds.shape}")
            # print(f"prefix_embeds shape: {prefix_embeds.shape}")

            # Prepend prefix embeddings to input embeddings
            # inputs_embeds: [batch_size, seq_len, hidden_size]
            # prefix_embeds: [batch_size, num_virtual_tokens, hidden_size]
            # result: [batch_size, num_virtual_tokens + seq_len, hidden_size]
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
            # print(f"inputs_embeds shape after concatenation: {inputs_embeds.shape}")
            # Expand attention mask to account for prefix tokens
            # Original attention_mask: [batch_size, seq_len]
            # New attention_mask: [batch_size, num_virtual_tokens + seq_len]
            prefix_attention_mask = torch.ones(
                batch_size,
                self.num_virtual_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

            # Prepend ignore labels (-100) for prefix tokens only if labels is not None
            # labels: [batch_size, seq_len] -> [batch_size, num_virtual_tokens + seq_len]
            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, self.num_virtual_tokens),
                    -100,  # Ignore index for loss computation
                    dtype=labels.dtype,
                    device=labels.device,
                )
                labels = torch.cat([prefix_labels, labels], dim=1)
            # Ensure dtypes of inputs_embeds match model weights (to avoid float/half mismatch)
            # Determine the model's dtype from the first weight
            model_dtype = next(self.model.parameters()).dtype
            if inputs_embeds.dtype != model_dtype:
                inputs_embeds = inputs_embeds.to(dtype=model_dtype)
            # attention_mask and labels should both be long/int if not already (for transformers)
            if attention_mask is not None and attention_mask.dtype != torch.long:
                attention_mask = attention_mask.to(dtype=torch.long)
            if labels is not None and labels.dtype != torch.long:
                labels = labels.to(dtype=torch.long)

            model_output = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            # Slice logits as before
            model_output.logits = model_output.logits[:, self.num_virtual_tokens :]
            return model_output
        else:
            return self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

    @abstractmethod
    def embed_tokens(self, batch_of_input_ids):
        pass

    @abstractmethod
    def get_embedding_dim(self):
        pass

    def generate(
        self,
        inputs_embeds,
        attention_mask,
        max_new_tokens=200,
        num_beams=20,
        do_sample=False,
        min_length=1,
        top_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
    ): # 9.56	13.73	8.1	13.2
        # NOTE: simply override the generate method for the diffusion llm
        if self.is_prefix_tuning_active:
            # Get batch size
            batch_size = inputs_embeds.shape[0]

            # Get prefix embeddings: [batch_size, num_virtual_tokens, hidden_size]
            prefix_embeds = self.prefix_tuning(batch_size)

            # Prepend prefix embeddings to input embeddings
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

            # Expand attention mask to account for prefix tokens
            prefix_attention_mask = torch.ones(
                batch_size,
                self.num_virtual_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

            # Ensure dtypes of inputs_embeds match model weights (to avoid float/half mismatch)
            model_dtype = next(self.model.parameters()).dtype
            if inputs_embeds.dtype != model_dtype:
                inputs_embeds = inputs_embeds.to(dtype=model_dtype)

            # Ensure attention_mask is long/int for transformers
            if attention_mask is not None and attention_mask.dtype != torch.long:
                attention_mask = attention_mask.to(dtype=torch.long)

            return self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )
        else:
            return self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )


@register_llm("llama-3-1-8b-instruct")
class Llama_3_1_8B_Instruct(LLM):
    def __init__(
        self,
        path_to_pretrained_model="./pretrained-llms/Llama-3.1-8B-Instruct/",
        freeze_layers=True,
        device="cuda:0",
    ):
        super().__init__(path_to_pretrained_model, freeze_layers, device)

    def get_embedding_dim(self):
        return self.model.model.embed_tokens.weight.shape[1]

    def embed_tokens(self, batch_of_input_ids):
        batch_of_input_ids = batch_of_input_ids
        return self.model.model.embed_tokens(batch_of_input_ids)

# === small import enhancement: add these inside your existing PEFT try block ===
try:
    from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA will not work.")


@register_llm("llada-8b-instruct")
class LLaDA_8B_Instruct(LLM):
    def __init__(
        self,
        path_to_pretrained_model="./pretrained-llms/LLaDA-8B-Instruct/",
        freeze_layers=True,
        device="cuda:0",
        local_files_only=True,
    ):
        # reuse parent logic to load base model + tokenizer
        super().__init__(path_to_pretrained_model, freeze_layers, device)
        # Ensure tokenizer is set (super() already sets it when model loaded)
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                path_to_pretrained_model,
                use_fast=True,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            self.tokenizer.pad_token_id = 126081

        # LLaDA-specific: mask token ID
        self.mask_token_id = 126336
        self.is_diffusion_llm = True

        self.mask_token_embedding = self.embed_tokens(
            torch.tensor([self.mask_token_id], device=device)
        )

    def get_embedding_dim(self):
        # matches pattern used in other classes
        return self.model.model.transformer.wte.weight.shape[1]

    def embed_tokens(self, batch_of_input_ids):
        return self.model.model.transformer.wte(batch_of_input_ids)

    def _add_gumbel_noise(self, logits, temperature):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves
        perplexity score but reduces generation quality. Thus, we use float64.
        """
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def _get_num_transfer_tokens(self, mask_index, steps):
        """
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
        the expected number of tokens transitioned at each step should be consistent.

        This function is designed to precompute the number of tokens that need to be
        transitioned at each step.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = (
            torch.zeros(
                mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
            )
            + base
        )

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1

        return num_transfer_tokens

    def _forward_process(self, inputs_embeds, eps=1e-3):
        b, l, d = inputs_embeds.shape
        t = torch.rand(b, device=inputs_embeds.device)
        p_mask = (1.0 - eps) * t + eps  # [B]
        p_mask = p_mask[:, None].repeat(1, l)
        masked_indices = torch.rand((b, l), device=inputs_embeds.device) < p_mask
        # self.mask_token_embedding is [1, D], need [B, L, D] for where operation
        mask_embedding_expanded = self.mask_token_embedding.unsqueeze(1).expand(b, l, d)
        noisy_embeds = torch.where(
            masked_indices.unsqueeze(-1), mask_embedding_expanded, inputs_embeds
        )
        # import matplotlib.pyplot as plt
        # plt.imshow(noisy_embeds[0].cpu().detach().numpy()[:, :100])
        # plt.savefig("noisy_embeds.png")
        # plt.close()
        # plt.imshow(inputs_embeds[0].cpu().detach().numpy()[:, :100])
        # plt.savefig("inputs_embeds.png")
        # plt.close()
        # exit()

        return noisy_embeds, masked_indices, p_mask

    def train_step(self, inputs_embeds, prompt_lengths, input_ids):
        noisy_embeds, masked_indices, p_mask = self._forward_process(inputs_embeds)
        token_positions = torch.arange(
            noisy_embeds.shape[1], device=noisy_embeds.device
        ).expand(noisy_embeds.size(0), noisy_embeds.size(1))
        prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
        # import matplotlib.pyplot as plt
        # plt.imshow(noisy_embeds[0].cpu().detach().numpy()[:, :100])
        # plt.savefig("noisy_embeds_before_correction.png")
        # plt.close()
        # noisy_embeds[prompt_mask] = inputs_embeds[prompt_mask]
        # plt.imshow(noisy_embeds[0].cpu().detach().numpy()[:, :100])
        # plt.savefig("noisy_embeds_after_correction.png")
        # plt.close()
        # exit()
        answer_lengths = torch.sum(
            (1 - prompt_mask.to(torch.int64)), dim=-1, keepdim=True
        )
        answer_lengths = answer_lengths.repeat(1, noisy_embeds.shape[1])
        # import matplotlib.pyplot as plt
        # plt.plot(masked_indices[0].cpu().detach().numpy()[:])
        # plt.savefig("masked_indices.png")
        # plt.close()
        # exit()
        model_output = self.model(
            inputs_embeds=noisy_embeds,
            attention_mask=torch.ones_like(noisy_embeds, dtype=torch.long),
        )
        logits = model_output.logits
        # print("logits", logits.shape)
        # print("inputs_embeds", inputs_embeds.shape)
        # print("masked_indices", masked_indices.shape)
        # print("p_mask", p_mask.shape)
        # print("answer_lengths", answer_lengths.shape)
        # exit()
        token_loss = (
            F.cross_entropy(
                logits[masked_indices], input_ids[masked_indices], reduction="none"
            )
            / p_mask[masked_indices]
        )
        ce_loss = (
            torch.sum(token_loss / answer_lengths[masked_indices].to(token_loss.dtype))
            / input_ids.shape[0]
        )
        model_output.loss = ce_loss

        # Get predictions
        pred_ids = torch.argmax(logits, dim=-1)

        # Get masked tokens GT and predictions as text (per sample in batch)
        masked_tokens_gt_text = []
        masked_tokens_pred_text = []
        for i in range(input_ids.shape[0]):
            gt_tokens = input_ids[i][masked_indices[i]]
            pred_tokens = pred_ids[i][masked_indices[i]]
            masked_tokens_gt_text.append(
                self.tokenizer.decode(gt_tokens, skip_special_tokens=True)
            )
            masked_tokens_pred_text.append(
                self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
            )

        # Get whole response GT and predictions as text (tokens after prompt)
        response_gt_text = []
        response_pred_text = []
        response_mask = ~prompt_mask
        for i in range(input_ids.shape[0]):
            gt_response_tokens = input_ids[i][response_mask[i]]
            pred_response_tokens = pred_ids[i][response_mask[i]]
            response_gt_text.append(
                self.tokenizer.decode(gt_response_tokens, skip_special_tokens=True)
            )
            response_pred_text.append(
                self.tokenizer.decode(pred_response_tokens, skip_special_tokens=True)
            )

        model_output.masked_tokens_gt_text = masked_tokens_gt_text
        model_output.masked_tokens_pred_text = masked_tokens_pred_text
        model_output.response_gt_text = response_gt_text
        model_output.response_pred_text = response_pred_text
        # print("response_gt_text", response_gt_text)
        # print("response_pred_text", response_pred_text)
        return model_output

    @torch.no_grad()
    def generate_in_one_pass(
        self, inputs_embeds, attention_mask=None, max_new_tokens=128, temperature=0.0
    ):
        return self.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            steps=1,
            gen_length=max_new_tokens,
            block_length=max_new_tokens,
            temperature=temperature,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds=None,
        attention_mask=None,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
        return_dict=False,
        **kwargs,
    ):
        # 13.973 14.142
        if inputs_embeds is None:
            raise ValueError("'inputs_embeds' must be provided.")

        # Prepare prompt embeddings
        prompt_embeddings = inputs_embeds.to(
            self.model.device, dtype=self.model.model.transformer.wte.weight.dtype
        )
        batch_size, prompt_length, _ = prompt_embeddings.shape

        # Token placeholder tensor (we do not have prompt token ids, so use 0 as a non-mask sentinel)
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=self.model.device,
        )
        x[:, :prompt_length] = 0

        # Extend attention mask if provided
        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask.to(device=self.model.device),
                    torch.ones(
                        (batch_size, gen_length),
                        dtype=attention_mask.dtype,
                        device=self.model.device,
                    ),
                ],
                dim=-1,
            )

        # Pre-compute mask token embedding
        mask_token_embedding = self.embed_tokens(
            torch.tensor([mask_id], device=self.model.device)
        )

        # Initialize embeddings: prompt + masked generation tokens
        sequence_embeds = torch.cat(
            [
                prompt_embeddings,
                mask_token_embedding.expand(batch_size, gen_length, -1),
            ],
            dim=1,
        )

        def _run_model_with_embeds(embeds, attn_mask):
            model_kwargs = {"inputs_embeds": embeds}
            if attn_mask is not None:
                model_kwargs["attention_mask"] = attn_mask
            return self.model(**model_kwargs).logits

        # Calculate number of tokens to unmask per step
        total_masked = gen_length
        tokens_per_step = max(1, total_masked // steps)

        for step_idx in range(steps):
            mask_index = x == mask_id
            num_masked = mask_index.sum(dim=-1)
            
            # If all tokens are unmasked, break early
            if num_masked.max() == 0:
                break

            if cfg_scale > 0.0:
                # Classifier-free guidance: mask out prompt embeddings for unconditional branch
                un_sequence_embeds = sequence_embeds.clone()
                un_sequence_embeds[:, :prompt_length] = mask_token_embedding.expand(
                    batch_size, prompt_length, -1
                )
                combined_embeds = torch.cat(
                    [sequence_embeds, un_sequence_embeds], dim=0
                )

                attn_mask_used = None
                if attention_mask is not None:
                    attn_mask_used = torch.cat(
                        [attention_mask, attention_mask], dim=0
                    )

                logits = _run_model_with_embeds(combined_embeds, attn_mask_used)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = _run_model_with_embeds(sequence_embeds, attention_mask)

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = self._add_gumbel_noise(
                logits, temperature=temperature
            )
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # Determine how many tokens to unmask this step
            if step_idx == steps - 1:
                # Last step: unmask all remaining
                num_to_unmask = num_masked
            else:
                # Calculate tokens to unmask progressively
                num_to_unmask = torch.clamp(
                    torch.tensor([tokens_per_step] * batch_size, device=x.device),
                    max=num_masked
                )

            transfer_index = torch.zeros_like(
                x0, dtype=torch.bool, device=x0.device
            )
            for j in range(confidence.shape[0]):
                k = min(int(num_to_unmask[j].item()), int(num_masked[j].item()))
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

            if transfer_index.any():
                new_token_embeds = self.embed_tokens(x0[transfer_index])
                sequence_embeds[transfer_index] = new_token_embeds

        if return_dict:
            return {"sequences": x}
        return x
