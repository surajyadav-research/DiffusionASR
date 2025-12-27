rsync -avzP -e "ssh -i ~/.ssh/puneet" \
    --include 'hindi__*/' \
    --include 'hindi__*/*pred*' \
    --include 'hindi__*/*gt*' \
    --include 'hindi__*/*best_model_no_loss.pt' \
    --include 'hindi__*/*.json' \
    --include 'hindi__*/*.log' \
    --exclude '*' \
    abrol@IP_ADDY:/home/abrol/puneet/speech-align-llm/checkpoints/ \
    /home/abrol/puneet/speech-align-llm/checkpoints/
