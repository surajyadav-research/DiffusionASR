# install libraries
# pip install transformers==4.41.2 peft==0.11.1 librosa==0.10.2

import transformers
import librosa

# load the model pipeline on gpu:0
pipe = transformers.pipeline(model="./benchmarks/checkpoints/shuka_v1/shuka-1", local_files_only=True, device=0, torch_dtype='bfloat16', trust_remote_code=True)


audio, sr = librosa.load("DATASETS/IndicSUPERB/kb_data_clean_wav/hindi/test/audio/536/844424930331319-536-m.wav", sr=16000)
turns = [
          {'role': 'system', 'content': 'Respond naturally and informatively.'},
          {'role': 'user', 'content': '<|audio|>'}
        ]

pipe({'audio': audio, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=512)
