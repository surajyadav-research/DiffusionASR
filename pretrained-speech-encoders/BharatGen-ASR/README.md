# Instructions for ESPNet Inference Environment Creation

- Start with the the espnet docker image `espnet/espnet:gpu-latest`
- Create a container with the required resources. It is essential that you increase the default shared memory to 8GB with `--shm-size 8GB`
- Inside the docker container, do `pip install espnet`
- Next, open file `/opt/miniconda/lib/python3.9/site-packages/espnet2/asr/espnet_model.py` and navigate to line `121` and change it from `from warprnnt_pytorch import RNNTLoss` to `from torchaudio.transforms import RNNTLoss` and go to line 125 and remove it.
```python
123    self.criterion_transducer = RNNTLoss(
124        blank=self.blank_id,
125        fastemit_lambda=0.0, # ---> Remove this line
126    )
``` 

Now you are ready to run the inference.py script.
To run inference
```bash
python inference.py --config branchformer/config.yaml --model_file branchformer/40epoch.pth  --audio test.wav
```
You might have to update paths in the branchformer/config.yaml as per your requirements for your changes.