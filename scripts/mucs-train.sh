# ./scripts/train.sh ./configs/mucs/mucs--openai-whisper-large-v3_resnet-large_krutrim-2-12B-instruct__prompted.yaml
# ./scripts/infer.sh ./configs/mucs/mucs--openai-whisper-large-v3_resnet-large_krutrim-2-12B-instruct__prompted.yaml

./scripts/train.sh ./configs/mucs/mucs--openai-whisper-large-v3_multi-res-conv-large_krutrim-2-12B-instruct__prompted.yaml
./scripts/infer.sh ./configs/mucs/mucs--openai-whisper-large-v3_multi-res-conv-large_krutrim-2-12B-instruct__prompted.yaml

./scripts/train.sh ./configs/mucs/mucs--openai-whisper-large-v3_multi-res-conv-xlarge_krutrim-2-12B-instruct__prompted.yaml
./scripts/infer.sh ./configs/mucs/mucs--openai-whisper-large-v3_multi-res-conv-xlarge_krutrim-2-12B-instruct__prompted.yaml

./scripts/train.sh ./configs/mucs/mucs--openai-whisper-large-v3_multi-res-conv-xxlarge_krutrim-2-12B-instruct__prompted.yaml
./scripts/infer.sh ./configs/mucs/mucs--openai-whisper-large-v3_multi-res-conv-xxlarge_krutrim-2-12B-instruct__prompted.yaml

./scripts/train.sh ./configs/mucs/mucs--openai-whisper-large-v3_linear-large-sln_krutrim-2-12B-instruct__prompted.yaml
./scripts/infer.sh ./configs/mucs/mucs--openai-whisper-large-v3_linear-large-sln_krutrim-2-12B-instruct__prompted.yaml