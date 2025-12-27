import argparse
import time

import soundfile as sf
import torch
from espnet2.bin.asr_inference import Speech2Text


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    audio = args.audio
    speech2text = Speech2Text(
        asr_train_config=args.config, 
        asr_model_file=args.model_file,
        device=device,
        beam_size=5
    )
    wav, rate = sf.read(audio)

    # resample to 16kHz
    if rate != 16000:
        import librosa
        wav = librosa.resample(wav, rate, 16000)
        rate = 16000

    # wav = wav[:16000*30]
    start = time.perf_counter()
    n_best = speech2text(wav)
    stop = time.perf_counter()
    print(n_best[0][0])
    print(f"Time taken: {stop-start:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Branchformer1024/config.yaml')
    parser.add_argument('--model_file', type=str, default='Branchformer1024/model.pth')
    parser.add_argument('--audio', type=str, required=True)
    args = parser.parse_args()
    main(args)
