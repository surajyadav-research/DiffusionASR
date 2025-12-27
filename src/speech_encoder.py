"""
Speech Encoder Module

To Extend:
    - Make a new class subclassing `SpeechEncoder` with the following methods:
        - `__init__`: Initialize the speech encoder
        - `encode_speech`: Encode the speech
        - `get_embedding_dim`: Get the dimension of the embedding
"""

from abc import ABC, abstractmethod

import torch
from transformers import AutoModel

SPEECH_ENCODER_REGISTRY = {}


def register_speech_encoder(name):
    """
    Decorator to register speech encoder classes in the SPEECH_ENCODER_REGISTRY dictionary.

    Args:
        name (str): The name of the speech encoder.
    """

    def wrapper(cls):
        SPEECH_ENCODER_REGISTRY[name] = cls
        return cls

    return wrapper


class SpeechEncoder(ABC, torch.nn.Module):
    def __init__(self, model_name: str = None, freeze_layers: bool = True):
        super().__init__()
        self.speech_encoder = None
        if model_name is not None:
            self._initialize_encoder(model_name)
            if freeze_layers:
                self._freeze_encoder()

    def _initialize_encoder(self, model_name: str):
        """Initialize the speech encoder model. Override this method for custom initialization."""
        self.speech_encoder = AutoModel.from_pretrained(model_name)

    def _freeze_encoder(self):
        """Freeze the encoder parameters if they exist."""
        if self.speech_encoder is not None:
            for param in self.speech_encoder.parameters():
                param.requires_grad = False

    @abstractmethod
    def encode_speech(
        self,
        audio=None,
        audio_padding_mask=None,
        audio_mel=None,
        audio_mel_padding_mask=None,
    ):
        pass

    @abstractmethod
    def get_embedding_dim(self):
        pass


def get_speech_encoder_class(name):
    """Get speech encoder class by name from registry"""
    if name not in SPEECH_ENCODER_REGISTRY:
        available_speech_encoders = list(SPEECH_ENCODER_REGISTRY.keys())
        raise ValueError(
            f"Unknown speech encoder class: {name}. Available options: {available_speech_encoders}"
        )
    return SPEECH_ENCODER_REGISTRY[name]


@register_speech_encoder("wavlm-large")
class WavLmSpeechEncoder(SpeechEncoder):
    def __init__(self, model_name="microsoft/wavlm-large", freeze_layers=True):
        super().__init__(model_name, freeze_layers)

    def get_embedding_dim(self):
        return self.speech_encoder.config.hidden_size

    def encode_speech(
        self, audio, audio_padding_mask, audio_mel=None, audio_mel_padding_mask=None
    ):
        attention_mask = 1 - audio_padding_mask
        # NOTE: audio_padding_mask is 1 for padded tokens, 0 for unpadded tokens
        # NOTE: and WavLM needs attention mask rather than padding mask
        speech_encoder_output = self.speech_encoder(audio, attention_mask)
        return speech_encoder_output.last_hidden_state


@register_speech_encoder("hubert-xlarge-ls960-ft-finetuned")
class HubertXlSpeechEncoderFinetuned(SpeechEncoder):
    def __init__(
        self, model_name="facebook/hubert-xlarge-ls960-ft", freeze_layers=True
    ):
        super().__init__(model_name, freeze_layers)

    def get_embedding_dim(self):
        return self.speech_encoder.config.hidden_size

    def encode_speech(
        self, audio, audio_padding_mask, audio_mel=None, audio_mel_padding_mask=None
    ):
        attention_mask = 1 - audio_padding_mask
        speech_encoder_output = self.speech_encoder(audio, attention_mask)
        return speech_encoder_output.last_hidden_state


@register_speech_encoder("mms-1b")
class MmsSpeechEncoder(SpeechEncoder):
    def __init__(self, model_name="facebook/mms-1b", freeze_layers=True):
        super().__init__(model_name, freeze_layers)

    def get_embedding_dim(self):
        return self.speech_encoder.config.hidden_size

    def encode_speech(
        self, audio, audio_padding_mask, audio_mel=None, audio_mel_padding_mask=None
    ):
        attention_mask = 1 - audio_padding_mask
        speech_encoder_output = self.speech_encoder(audio, attention_mask)
        return speech_encoder_output.last_hidden_state


@register_speech_encoder("indicwav2vec-hindi-finetuned")
class IndicWav2VecHindiSpeechEncoderFinetuned(SpeechEncoder):
    def __init__(self, model_name="ai4bharat/indicwav2vec-hindi", freeze_layers=True):
        super().__init__(model_name, freeze_layers)

    def get_embedding_dim(self):
        return self.speech_encoder.config.hidden_size

    def encode_speech(
        self, audio, audio_padding_mask, audio_mel=None, audio_mel_padding_mask=None
    ):
        attention_mask = 1 - audio_padding_mask
        speech_encoder_output = self.speech_encoder(audio, attention_mask)
        return speech_encoder_output.last_hidden_state


@register_speech_encoder("indicwhisper-large")
class IndicWhisperLargeSpeechEncoder(SpeechEncoder):
    def __init__(
        self,
        model_name="./pretrained-speech-encoders/IndicWhisper/hindi_models/whisper-large-hi-noldcil",
        freeze_layers=True,
    ):
        super().__init__(model_name, freeze_layers)

    def get_embedding_dim(self):
        return self.speech_encoder.config.hidden_size

    def encode_speech(
        self, audio, audio_padding_mask, audio_mel=None, audio_mel_padding_mask=None
    ):
        attention_mask = 1 - audio_padding_mask
        speech_encoder_output = self.speech_encoder(audio, attention_mask)
        return speech_encoder_output.last_hidden_state


@register_speech_encoder("indicwav2vec-base")
class IndicWav2VecBaseSpeechEncoder(SpeechEncoder):
    def __init__(
        self,
        path_to_model_weights="./pretrained-speech-encoders/IndicWav2Vec-base/indicw2v_base_pretrained.pt",
        freeze_layers=True,
    ):
        super().__init__(
            None, freeze_layers
        )  # Pass None to skip default initialization
        self._initialize_encoder(path_to_model_weights)
        if freeze_layers:
            self._freeze_encoder()

    def _initialize_encoder(self, path_to_model_weights: str):
        from fairseq import checkpoint_utils

        model, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [path_to_model_weights]
        )
        self.speech_encoder = model[0]

    def get_embedding_dim(self):
        return 768

    def encode_speech(
        self, audio, audio_padding_mask, audio_mel=None, audio_mel_padding_mask=None
    ):
        # unlike the huggingface models, this model needs a padding mask and not attention mask
        speech_encoder_output = self.speech_encoder.extract_features(
            audio, audio_padding_mask
        )
        speech_encoder_output = speech_encoder_output["x"]
        return speech_encoder_output


@register_speech_encoder("indic-mr-hubert-large")
class IndicMrHubertLargeSpeechEncoder(SpeechEncoder):
    def __init__(
        self,
        path_to_model_weights="./pretrained-speech-encoders/Indic-MR-HuBERT/0/checkpoints/checkpoints/checkpoint_best.pt",
        freeze_layers=True,
    ):
        super().__init__(None, freeze_layers)
        self._initialize_encoder(path_to_model_weights)
        if freeze_layers:
            self._freeze_encoder()

    def _initialize_encoder(self, path_to_model_weights: str):
        from fairseq import checkpoint_utils

        model, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [path_to_model_weights]
        )
        self.speech_encoder = model[0]

    def get_embedding_dim(self):
        return 1024

    def encode_speech(
        self, audio, audio_padding_mask, audio_mel=None, audio_mel_padding_mask=None
    ):
        # unlike the huggingface models, this model needs a padding mask and not attention mask
        speech_encoder_output, _ = self.speech_encoder.extract_features(
            audio, audio_padding_mask, last_layer=True
        )
        return speech_encoder_output


@register_speech_encoder("clsril-23")
class Clsril23SpeechEncoder(SpeechEncoder):
    def __init__(
        self,
        path_to_model_weights="./pretrained-speech-encoders/CLSRIL-23/CLSRIL-23.pt",
        freeze_layers=True,
    ):
        super().__init__(None, freeze_layers)
        self._initialize_encoder(path_to_model_weights)
        if freeze_layers:
            self._freeze_encoder()

    def _initialize_encoder(self, path_to_model_weights: str):
        from fairseq import checkpoint_utils

        model, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [path_to_model_weights]
        )
        self.speech_encoder = model[0]

    def get_embedding_dim(self):
        return 768

    def encode_speech(
        self, audio, audio_padding_mask, audio_mel=None, audio_mel_padding_mask=None
    ):
        # unlike the huggingface models, this model needs a padding mask and not attention mask
        speech_encoder_output = self.speech_encoder.extract_features(
            audio, audio_padding_mask
        )
        return speech_encoder_output["x"]


@register_speech_encoder("clsril-23-hindi-finetuned")
class Clsril23HindiFinetunedSpeechEncoder(SpeechEncoder):
    def __init__(
        self,
        model_name="Harveenchadha/vakyansh-wav2vec2-hindi-him-4200",
        freeze_layers=True,
    ):
        super().__init__(model_name, freeze_layers)

    def get_embedding_dim(self):
        return 768

    def encode_speech(
        self, audio, audio_padding_mask, audio_mel=None, audio_mel_padding_mask=None
    ):
        attention_mask = 1 - audio_padding_mask
        speech_encoder_output = self.speech_encoder(audio, attention_mask)
        return speech_encoder_output.last_hidden_state


@register_speech_encoder("ekstep-hindi-4kh")
class HindiPretrained4khSpeechEncoder(SpeechEncoder):
    def __init__(
        self,
        path_to_model_weights="./pretrained-speech-encoders/hindi_pretrained_4kh/hindi_pretrained_4kh.pt",
        freeze_layers=True,
    ):
        super().__init__(None, freeze_layers)
        self._initialize_encoder(path_to_model_weights)
        if freeze_layers:
            self._freeze_encoder()

    def _initialize_encoder(self, path_to_model_weights: str):
        from fairseq import checkpoint_utils

        model, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [path_to_model_weights]
        )
        self.speech_encoder = model[0]

    def get_embedding_dim(self):
        return 768

    def encode_speech(
        self, audio, audio_padding_mask, audio_mel=None, audio_mel_padding_mask=None
    ):
        # unlike the huggingface models, this model needs a padding mask and not attention mask
        speech_encoder_output = self.speech_encoder.extract_features(
            audio, audio_padding_mask
        )
        return speech_encoder_output["x"]


@register_speech_encoder("openai-whisper-large-v3")
class OpenaiWhisperLargeV3SpeechEncoder(SpeechEncoder):
    def __init__(self, model_name="openai/whisper-large-v3", freeze_layers=True):
        super().__init__(model_name, freeze_layers)
        self.speech_encoder = self.speech_encoder.encoder

    def get_embedding_dim(self):
        return 1280

    def encode_speech(
        self, audio_mel, audio_mel_padding_mask, audio=None, audio_padding_mask=None
    ):
        # attention mask is ignored in whisper, audio_mel_padding_mask is passed for the sake of consistency
        # NOTE: https://github.com/huggingface/transformers/blob/d7b87b415a5dd4a3152051e1a0abd098a02c5bfa/src/transformers/models/whisper/modeling_whisper.py#L656
        speech_encoder_output = self.speech_encoder(
            input_features=audio_mel, attention_mask=audio_mel_padding_mask
        )
        return speech_encoder_output[0]


@register_speech_encoder("openai-whisper-large-v2")
class OpenaiWhisperLargeV2SpeechEncoder(SpeechEncoder):
    def __init__(self, model_name="openai/whisper-large-v2", freeze_layers=True):
        super().__init__(model_name, freeze_layers)
        self.speech_encoder = self.speech_encoder.encoder

    def get_embedding_dim(self):
        return 1280

    def encode_speech(
        self, audio_mel, audio_mel_padding_mask, audio=None, audio_padding_mask=None
    ):
        speech_encoder_output = self.speech_encoder(
            input_features=audio_mel, attention_mask=audio_mel_padding_mask
        )
        return speech_encoder_output[0]


@register_speech_encoder("bharatgen-asr-branchformer")
class BharatGenAsrBranchformerSpeechEncoder(SpeechEncoder):
    def __init__(
        self,
        path_to_model_weights="./pretrained-speech-encoders/BharatGen-ASR/Branchformer1024",
        freeze_layers=True,
    ):
        super().__init__(model_name=path_to_model_weights, freeze_layers=freeze_layers)

    def _initialize_encoder(self, path_to_model_weights: str):
        import os

        from espnet2.bin.asr_inference import Speech2Text

        # Construct paths to config and model files
        config_path = os.path.join(path_to_model_weights, "config.yaml")
        model_path = os.path.join(path_to_model_weights, "model.pth")

        self.speech_encoder = Speech2Text(
            asr_train_config=config_path,
            asr_model_file=model_path,
            device="cpu",  # SpeechEncoder2Adapter2Llm class will transfer the model to the appropriate device
            beam_size=1,
        ).asr_model.encoder

    def get_embedding_dim(self):
        return 256

    def encode_speech(
        self, audio, audio_padding_mask, audio_mel=None, audio_mel_padding_mask=None
    ):
        audio_mel = audio_mel.permute(0, 2, 1)  # [B, F, T] -> [B, T, F]
        # NOTE: audio_mel_padding_mask is 1 for padded tokens, 0 for unpadded tokens;
        audio_mel_mask = 1 - audio_mel_padding_mask
        speech_lengths = audio_mel_mask.sum(dim=1).int()
        encoder_out = self.speech_encoder(audio_mel, speech_lengths)

        return encoder_out[0]


@register_speech_encoder("openai-whisper-large-v3-interim-outputs")
class OpenaiWhisperLargeV3InterimOutputsSpeechEncoder(SpeechEncoder):
    def __init__(self, model_name="openai/whisper-large-v3", freeze_layers=True):
        super().__init__(model_name, freeze_layers)
        self.speech_encoder = self.speech_encoder.encoder

    def get_embedding_dim(self):
        return 1280

    def encode_speech(
        self, audio_mel, audio_mel_padding_mask, audio=None, audio_padding_mask=None
    ):
        # attention mask is ignored in whisper, audio_mel_padding_mask is passed for the sake of consistency
        # NOTE: https://github.com/huggingface/transformers/blob/d7b87b415a5dd4a3152051e1a0abd098a02c5bfa/src/transformers/models/whisper/modeling_whisper.py#L656
        speech_encoder_output = self.speech_encoder(
            input_features=audio_mel,
            attention_mask=audio_mel_padding_mask,
            output_hidden_states=True,
        )
        # GOTCHA: this is a list of tensors, each tensor is of shape [B, L, D]
        # HOTFIX: this breaks the backward compatibility of the speech encoder
        # NOTE: we only consider the last 5 blocks of the speech encoder
        return torch.stack(speech_encoder_output.hidden_states, dim=1)  # [B, 33, L, D]

    def get_num_blocks(self):
        return 33

    def get_num_mels(self):
        return 128


@register_speech_encoder("spring-inx-data2vec-hindi-finetuned")
class SpringInxData2VecSpeechEncoderFinetuned(SpeechEncoder):
    def __init__(
        self,
        model_name="./pretrained-speech-encoders/spring-inx-models/SPRING_INX_data2vec_Hindi.pt",
        freeze_layers=True,
    ):
        super().__init__(None, freeze_layers)
        self._initialize_encoder(model_name)
        if freeze_layers:
            self._freeze_encoder()

    def _initialize_encoder(self, path_to_model_weights: str):
        from fairseq import checkpoint_utils

        model, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [path_to_model_weights]
        )
        self.speech_encoder = model[0]

    def get_embedding_dim(self):
        return 128

    def encode_speech(
        self, audio, audio_padding_mask, audio_mel=None, audio_mel_padding_mask=None
    ):
        speech_encoder_output = self.speech_encoder.extract_features(
            audio, audio_padding_mask
        )
        return speech_encoder_output["x"]
