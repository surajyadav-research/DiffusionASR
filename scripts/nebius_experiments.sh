# ./scripts/train.sh ./configs/hindi/openai-whisper-large-v3_linear-large_krutrim-2-12b-instruct.yaml
# ./scripts/infer.sh ./configs/hindi/openai-whisper-large-v3_linear-large_krutrim-2-12b-instruct.yaml
# ./scripts/train.sh ./configs/hindi/openai-whisper-large-v3_convolutional-projector_krutrim-2-12b-instruct.yaml
# ./scripts/infer.sh ./configs/hindi/openai-whisper-large-v3_convolutional-projector_krutrim-2-12b-instruct.yaml
./scripts/train.sh ./configs/hindi/openai-whisper-large-v3_resnet-projector_krutrim-2-12b-instruct.yaml
./scripts/infer.sh ./configs/hindi/openai-whisper-large-v3_resnet-projector_krutrim-2-12b-instruct.yaml
# ./scripts/train.sh ./configs/hindi/openai-whisper-large-v3_qformer-small-projector_krutrim-2-12b-instruct.yaml # running on precision-h100 machine
# ./scripts/infer.sh ./configs/hindi/openai-whisper-large-v3_qformer-small-projector_krutrim-2-12b-instruct.yaml
./scripts/train.sh ./configs/hindi/openai-whisper-large-v3_linear-large-projector-ln_krutrim-2-12b-instruct.yaml
./scripts/infer.sh ./configs/hindi/openai-whisper-large-v3_linear-large-projector-ln_krutrim-2-12b-instruct.yaml