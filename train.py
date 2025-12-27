import argparse
import logging
import os
from datetime import datetime
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

import wandb
from src.dataset import AsrDataset4LLaDA
from src.model import SpeechEncoder2Adapter2Llm
from src.utils import (
    batch_to_device,
    get_model_size,
    load_config,
    save_model,
    set_random_seed,
)


# CONFIG AND ARGS
def parse_args():
    parser = argparse.ArgumentParser(description="Train ASR model")
    parser.add_argument(
        "--path_to_config_file",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    return parser.parse_args()


args = parse_args()
config = load_config(args.path_to_config_file)
set_random_seed()

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, config["paths"]["checkpoints_dir"])
ARTIFACTS_DIR = os.path.join(BASE_DIR, config["paths"]["artifacts_dir"])
WANDB_DIR = os.path.join(BASE_DIR, config["paths"]["wandb_dir"])
DATA_DIR = os.path.join(BASE_DIR, config["paths"]["data_dir"])
os.makedirs(WANDB_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(os.path.join(CHECKPOINTS_DIR, config["wandb"]["name"]), exist_ok=True)

# ENV
for key, value in config["env"].items():
    os.environ[key.upper()] = str(value)
os.environ["WANDB_DIR"] = WANDB_DIR

# WANDB AND LOGGING
wandb.init(
    project=config["wandb"]["project"],
    entity=config["wandb"]["entity"],
    name=config["wandb"]["name"],
    config=config,
)

log_file = os.path.join(
    wandb.run.dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file)],
)
logger = logging.getLogger(__name__)


def main():
    # RESUME FROM CHECKPOINT
    if config["training"].get("resume_from_checkpoint", False):
        list_of_paths_to_checkpoints = glob(
            os.path.join(
                CHECKPOINTS_DIR, config["wandb"]["name"], "best_model_val_loss_*.pt"
            )
        )
        if len(list_of_paths_to_checkpoints) == 0:
            logger.error("No checkpoints found")
            print("ERROR: No checkpoints found to resume from")
            exit(1)
        else:
            print("list_of_paths_to_checkpoints", list_of_paths_to_checkpoints)
            path_to_best_checkpoint = sorted(
                list_of_paths_to_checkpoints,
                key=lambda x: float(x.split("_")[-1].split(".")[0]),
            )[0]
            print("Resuming from checkpoint: ", path_to_best_checkpoint)
        logger.info(f"Resuming from checkpoint: {path_to_best_checkpoint}")
    else:
        path_to_best_checkpoint = None
        logger.info("Training adapter from scratch")

    # LoRA adapter checkpoint logic
    lora_cfg = config["training"].get("lora_config", None)
    lora_adapter_ckpt = None
    if lora_cfg is not None:
        lora_adapter_ckpt_path = os.path.join(
            CHECKPOINTS_DIR, config["wandb"]["name"], "lora_adapter_checkpoint.pt"
        )
        if os.path.exists(lora_adapter_ckpt_path):
            print(f"Found LoRA adapter checkpoint at {lora_adapter_ckpt_path}")
            lora_adapter_ckpt = lora_adapter_ckpt_path
        else:
            print(
                f"No LoRA adapter checkpoint found at {lora_adapter_ckpt_path}, starting fresh."
            )

    # Prefix-tuning adapter checkpoint logic
    prefix_tuning_cfg = config["training"].get("prefix_tuning_config", None)
    prefix_tuning_adapter_ckpt = None
    if prefix_tuning_cfg is not None:
        prefix_tuning_adapter_ckpt_path = os.path.join(
            CHECKPOINTS_DIR,
            config["wandb"]["name"],
            "prefix_tuning_adapter_checkpoint.pt",
        )
        if os.path.exists(prefix_tuning_adapter_ckpt_path):
            print(
                f"Found Prefix-tuning adapter checkpoint at {prefix_tuning_adapter_ckpt_path}"
            )
            prefix_tuning_adapter_ckpt = prefix_tuning_adapter_ckpt_path
        else:
            print(
                f"No Prefix-tuning adapter checkpoint found at {prefix_tuning_adapter_ckpt_path}, starting fresh."
            )

    model = SpeechEncoder2Adapter2Llm(
        speech_encoder_name=config["model"]["speech_encoder_name"],
        llm_name=config["model"]["llm_name"],
        adapter_name=config["model"]["adapter_name"],
        device="cuda:0",
        path_to_adapter_weights=path_to_best_checkpoint,
        lora_config_dict=lora_cfg,
        lora_adapter_ckpt_path=lora_adapter_ckpt,
        prefix_tuning_config_dict=prefix_tuning_cfg,
        prefix_tuning_adapter_ckpt_path=prefix_tuning_adapter_ckpt,
    )

    train_dataset = AsrDataset4LLaDA(
        tokenizer=model.tokenizer,
        prompt=config["model"]["prompt"],
        mel_size=config["model"]["mel_size"],
        fix_length_audio=config["model"]["fix_length_audio"],
        fix_audio_duration=config["model"]["fix_audio_duration"],
        inference_mode=False,
        normalize=config["model"]["normalize"],
        input_type=config["model"]["input_type"],
        path_to_jsonl_file=os.path.join(DATA_DIR, config["paths"]["train_data"]),
        target_column=config["model"]["target_column"],
        llm_name=config["model"]["llm_name"],
    )

    valid_dataset = AsrDataset4LLaDA(
        tokenizer=model.tokenizer,
        prompt=config["model"]["prompt"],
        mel_size=config["model"]["mel_size"],
        fix_length_audio=config["model"]["fix_length_audio"],
        fix_audio_duration=config["model"]["fix_audio_duration"],
        inference_mode=False,
        normalize=config["model"]["normalize"],
        input_type=config["model"]["input_type"],
        target_column=config["model"]["target_column"],
        path_to_jsonl_file=os.path.join(DATA_DIR, config["paths"]["valid_data"]),
        llm_name=config["model"]["llm_name"],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=True,
        pin_memory=True,
        collate_fn=train_dataset.collator,
        drop_last=False,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=False,
        pin_memory=True,
        collate_fn=valid_dataset.collator,
        drop_last=False,
    )

    model_size = get_model_size(model)
    print(f"Model has {model_size} Million trainable parameters")

    # TRAINING PARAMETERS AND SETUP
    num_epochs = config["training"]["num_epochs"]
    num_steps = config["training"]["num_steps"]
    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    total_steps = num_epochs * len(train_dataloader)
    warmup_steps = config["training"]["warmup_steps"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    validation_frequency = config["training"]["validation_frequency"]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            min(step / warmup_steps, 1)
            if step < warmup_steps
            else max(
                0.0,
                1 - (step - warmup_steps) / (total_steps - warmup_steps),
            )
        ),
    )
    grad_scaler = torch.amp.GradScaler()

    logger.info(
        f"Model: {model.__class__.__name__} with {model.speech_encoder.__class__.__name__} -> {model.adapter.__class__.__name__} -> {model.llm.__class__.__name__}"
    )
    logger.info("Starting training with configuration:")
    logger.info(f"Number of epochs: {num_epochs} # whichever occurs first")
    logger.info(f"Number of steps: {num_steps} # whichever occurs first")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Target column: {config['model']['target_column']}")
    logger.info(f"Validation frequency: {validation_frequency}")

    print("num_epochs", num_epochs)
    print("num_steps", num_steps)
    print("gradient_accumulation_steps", gradient_accumulation_steps)
    print("warmup_steps", warmup_steps)
    print("total_steps", total_steps)
    print("target_column", config["model"]["target_column"])
    print("validation_frequency", validation_frequency)

    # NOTE: optional config pararms for special training runs; kept optional for backward compatibility
    if config["training"].get("resume_from_checkpoint", False):
        print(
            "resume_from_checkpoint",
            config["training"]["resume_from_checkpoint"],
            "| path_to_best_checkpoint",
            path_to_best_checkpoint,
        )
    if lora_cfg is not None:
        print("Using LLM LoRA")
    else:
        print("Not using LLM LoRA")

    if prefix_tuning_cfg is not None:
        print("Using LLM Prefix-tuning")
    else:
        print("Not using LLM Prefix-tuning")

    if config["training"].get("use_matching_loss", False):
        print("Using matching loss")
    else:
        print("Not using matching loss")

    list_of_train_combined_losses_per_batch = []
    list_of_train_lm_losses_per_batch = []
    list_of_train_matching_losses_per_batch = []
    list_of_train_accuracies_per_batch = []

    list_of_valid_combined_losses = []
    list_of_valid_lm_losses = []
    list_of_valid_matching_losses = []
    list_of_valid_accuracies = []

    best_val_loss = float("inf")

    # TRAINING LOOP
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        total_train_combined_loss = 0
        total_train_lm_loss = 0
        total_train_matching_loss = 0
        total_train_accuracy = 0

        total_val_combined_loss = 0
        total_val_lm_loss = 0
        total_val_matching_loss = 0
        total_val_accuracy = 0

        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        total_steps_in_epoch = len(train_dataloader) // gradient_accumulation_steps

        for step, batch in enumerate(train_pbar):
            # Check if we've reached the maximum number of steps
            current_step = epoch * len(train_dataloader) + step
            if current_step >= num_steps:
                logger.info(f"Completed {num_steps} steps")
                break

            model.train()
            batch = batch_to_device(batch, "cuda:0")

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs, matching_loss, accuracy = model(
                    batch,
                    use_matching_loss=config["training"].get(
                        "use_matching_loss", False
                    ),
                )

                lm_loss = outputs.loss
                if lm_loss is None:
                    # LLaDA models do not return an internal CE loss; compute it manually
                    shift_logits = outputs.logits[:, :-1, :].contiguous()
                    shift_labels = batch["labels"][:, 1:].contiguous()
                    lm_loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)).float(),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                combined_loss = lm_loss + matching_loss

                combined_loss = combined_loss / gradient_accumulation_steps
                lm_loss = lm_loss / gradient_accumulation_steps
                matching_loss = matching_loss / gradient_accumulation_steps

                list_of_train_lm_losses_per_batch.append(
                    lm_loss.detach().cpu().float().item()
                )
                list_of_train_matching_losses_per_batch.append(
                    matching_loss.detach().cpu().float().item()
                )
                list_of_train_combined_losses_per_batch.append(
                    combined_loss.detach().cpu().float().item()
                )
                list_of_train_accuracies_per_batch.append(
                    accuracy.detach().cpu().float().item()
                )

                # Log batch-level metrics
                current_lr = lr_scheduler.get_last_lr()[0]
                logger.debug(
                    f"Step {step}: LM Loss={lm_loss.item():.4f}, Matching Loss={matching_loss.item():.4f}, Accuracy={accuracy.item():.4f}, LR={current_lr:.6f}"
                )

                wandb.log(
                    {
                        "train/train_lm_loss_this_batch": lm_loss.detach()
                        .cpu()
                        .float()
                        .item(),
                        "train/train_matching_loss_this_batch": matching_loss.detach()
                        .cpu()
                        .float()
                        .item(),
                        "train/train_loss_this_batch": combined_loss.detach()  # retaining train_loss_this_batch for backward compatibility
                        .cpu()
                        .float()
                        .item(),
                        "train/train_accuracy_this_batch": accuracy.detach()
                        .cpu()
                        .float()
                        .item(),
                        "train/learning_rate": current_lr,
                    },
                    step=epoch * total_steps_in_epoch + step,
                )

                total_train_combined_loss += combined_loss.detach().cpu().float()
                total_train_lm_loss += lm_loss.detach().cpu().float()
                total_train_matching_loss += matching_loss.detach().cpu().float()
                total_train_accuracy += accuracy

                grad_scaler.scale(combined_loss).backward()

                if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            train_pbar.set_postfix(
                {
                    "train_loss": f"{total_train_combined_loss / (step + 1):.3f}",  # retaining train_loss for backward compatibility
                    "train_lm_loss": f"{total_train_lm_loss / (step + 1):.3f}",
                    "train_matching_loss": f"{total_train_matching_loss / (step + 1):.3f}",
                    "valid_loss": f"{total_val_combined_loss / len(valid_dataloader):.3f}",  # retaining valid_loss for backward compatibility
                    "valid_lm_loss": f"{total_val_lm_loss / len(valid_dataloader):.3f}",
                    "valid_matching_loss": f"{total_val_matching_loss / len(valid_dataloader):.3f}",  # retaining valid_matching_loss for backward compatibility
                    "train_accuracy": f"{total_train_accuracy / (step + 1):.3f}",
                    "valid_accuracy": f"{total_val_accuracy / len(valid_dataloader):.3f}",  # retaining valid_accuracy for backward compatibility
                }
            )
            num_samples_processed = (step + 1) * config["training"]["batch_size"]
            if num_samples_processed % validation_frequency == 0:
                print("Running validation at step", step + 1)
                logger.info(f"Running validation at step {step + 1}")
                model.eval()

                total_val_combined_loss = 0
                total_val_lm_loss = 0
                total_val_matching_loss = 0
                total_val_accuracy = 0

                list_of_valid_matching_losses = []
                list_of_valid_lm_losses = []
                list_of_valid_combined_losses = []
                list_of_valid_accuracies = []

                valid_pbar = tqdm(
                    valid_dataloader, desc=f"Validation @ Step {step + 1}", leave=False
                )

                # VALIDATION LOOP
                with torch.no_grad():
                    for batch in valid_pbar:
                        batch = batch_to_device(batch, "cuda:0")
                        with torch.amp.autocast(
                            device_type="cuda", dtype=torch.bfloat16
                        ):
                            outputs, matching_loss, accuracy = model(batch)
                            lm_loss = outputs.loss
                            if lm_loss is None:
                                # LLaDA models do not return an internal CE loss; compute it manually
                                shift_logits = outputs.logits[:, :-1, :].contiguous()
                                shift_labels = batch["labels"][:, 1:].contiguous()
                                lm_loss = torch.nn.functional.cross_entropy(
                                    shift_logits.view(
                                        -1, shift_logits.size(-1)
                                    ).float(),
                                    shift_labels.view(-1),
                                    ignore_index=-100,
                                )
                            combined_loss = lm_loss + matching_loss
                            total_val_combined_loss += (
                                combined_loss.detach().cpu().float()
                            )
                            total_val_lm_loss += lm_loss.detach().cpu().float()
                            total_val_matching_loss += (
                                matching_loss.detach().cpu().float()
                            )
                            total_val_accuracy += accuracy
                            list_of_valid_combined_losses.append(
                                combined_loss.detach().cpu().float().item()
                            )
                            list_of_valid_lm_losses.append(
                                lm_loss.detach().cpu().float().item()
                            )
                            list_of_valid_matching_losses.append(
                                matching_loss.detach().cpu().float().item()
                            )
                            list_of_valid_accuracies.append(
                                accuracy.detach().cpu().float().item()
                            )
                            list_of_valid_matching_losses.append(
                                matching_loss.detach().cpu().float().item()
                            )
                            # Log validation metrics
                            logger.debug(
                                f"Validation - Loss={combined_loss.item():.4f}, LM Loss={lm_loss.item():.4f}, Matching Loss={matching_loss.item():.4f}, Accuracy={accuracy.item():.4f}"
                            )

                    valid_pbar.set_postfix(
                        {
                            "valid_loss": f"{total_val_combined_loss / len(valid_dataloader):.3f}",  # retaining valid_loss for backward compatibility
                            "valid_lm_loss": f"{total_val_lm_loss / len(valid_dataloader):.3f}",
                            "valid_matching_loss": f"{total_val_matching_loss / len(valid_dataloader):.3f}",
                            "valid_accuracy": f"{total_val_accuracy / len(valid_dataloader):.3f}",  # retaining valid_accuracy for backward compatibility
                        }
                    )

                    # Log final validation metrics after all batches
                    avg_val_combined_loss = total_val_combined_loss / len(
                        valid_dataloader
                    )
                    avg_val_lm_loss = total_val_lm_loss / len(valid_dataloader)
                    avg_val_matching_loss = total_val_matching_loss / len(
                        valid_dataloader
                    )
                    avg_val_accuracy = total_val_accuracy / len(valid_dataloader)

                    # Save model if validation loss is better than previous best
                    if avg_val_combined_loss < best_val_loss:
                        best_val_loss = avg_val_combined_loss
                        checkpoint_path = os.path.join(
                            CHECKPOINTS_DIR,
                            config["wandb"]["name"],
                            f"best_model_val_loss_{best_val_loss:.4f}.pt",
                        )
                        save_model(model.adapter, optimizer, checkpoint_path)
                        path_to_best_model_no_loss = os.path.join(
                            CHECKPOINTS_DIR,
                            config["wandb"]["name"],
                            f"best_model_no_loss.pt",
                        )
                        save_model(model.adapter, optimizer, path_to_best_model_no_loss)
                        logger.info(
                            f"New best validation loss: {best_val_loss:.4f}. Model saved to {checkpoint_path}"
                        )
                        wandb.log(
                            {
                                "valid/best_validation_loss": best_val_loss,
                            },
                            step=epoch * total_steps_in_epoch + step,
                        )
                        # --- SAVE LORA ADAPTER WEIGHTS IF USING LORA ---
                        if lora_cfg is not None:
                            lora_adapter_ckpt = os.path.join(
                                CHECKPOINTS_DIR,
                                config["wandb"]["name"],
                                "lora_adapter_checkpoint.pt",
                            )
                            print(f"Saving LoRA adapter weights to {lora_adapter_ckpt}")
                            model.save_lora_adapter(lora_adapter_ckpt)

                        # --- SAVE PREFIX-TUNING ADAPTER WEIGHTS IF USING PREFIX-TUNING ---
                        if prefix_tuning_cfg is not None:
                            prefix_tuning_adapter_ckpt = os.path.join(
                                CHECKPOINTS_DIR,
                                config["wandb"]["name"],
                                "prefix_tuning_adapter_checkpoint.pt",
                            )
                            print(
                                f"Saving Prefix-tuning adapter weights to {prefix_tuning_adapter_ckpt}"
                            )
                            model.save_prefix_tuning_adapter(prefix_tuning_adapter_ckpt)

                    wandb.log(
                        {
                            "valid/valid_loss": avg_val_combined_loss.item(),  # retaining valid_loss for backward compatibility
                            "valid/valid_lm_loss": avg_val_lm_loss.item(),
                            "valid/valid_matching_loss": avg_val_matching_loss.item(),
                            "valid/valid_accuracy": avg_val_accuracy.item(),
                        },
                        step=epoch * total_steps_in_epoch + step,
                    )

                model.train()
                logger.info(
                    f"Validation completed - Loss: {avg_val_combined_loss:.4f}, LM Loss: {avg_val_lm_loss:.4f}, Matching Loss: {avg_val_matching_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}"
                )

        logger.info(f"Completed epoch {epoch + 1}/{num_epochs}")
        logger.info(
            f"Epoch {epoch + 1} metrics - Train Loss: {total_train_combined_loss/len(train_dataloader):.4f}, Train Accuracy: {total_train_accuracy/len(train_dataloader):.4f}"
        )


if __name__ == "__main__":
    main()
