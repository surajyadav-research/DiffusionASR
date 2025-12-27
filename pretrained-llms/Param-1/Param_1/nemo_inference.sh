export HYDRA_FULL_ERROR=1
python3 /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_eval.py \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    tensor_model_parallel_size=1 \
    pipeline_model_parallel_size=1 \
    inference.greedy=False \
    inference.tokens_to_generate=100 \
    inference.top_k=50 \
    inference.top_p=0.9 \
    inference.temperature=0.7 \
    inference.add_BOS=True \
    gpt_model_file="path_to_nemo_file" \
    prompts="दिवाली एक त्यौहार है"


# we can add more prompts to the prompts and run the script again to get more responses
#megatron_gpt_sft_full_2.5T2.5T_epoch3.nemo





