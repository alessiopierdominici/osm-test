import torch
import os
import clip
import argparse
from tqdm import tqdm
from create_model import _load_osm_clip
from utils.dataset_utils import load_datasets_as_single_one
from utils.general_utils import (
    convert_models_to_fp32, 
    get_formatted_datetime, 
    inference_with_model, 
    initialize_deterministic_mode, 
    save_checkpoint
)
from utils.training_utils import (
    load_config,
    setup_logging,
    setup_optimizer,
    setup_device,
    calculate_effective_batch_size,
    validate_common_config,
    log_training_config
)
import wandb
from utils.logging_utils import initialize_wandb_logging
import logging


def run_epoch(model, dataloader, optimizer, loss_img, loss_txt, device, config, mode="train"):
    """Run one epoch of training or validation."""
    epoch_loss = 0
    model.train() if mode == "train" else model.eval()
    
    use_grad_accum = config['gradient_accumulation']['enabled']
    accum_steps = config['gradient_accumulation']['steps']
    use_original_clip = config['model']['use_original_clip']
    
    if use_grad_accum and mode == "train":
        optimizer.zero_grad()
        accumulation_step = 0

    for batch in tqdm(dataloader, desc=f"{mode.capitalize()}"):
        images, texts = batch
        texts = torch.cat(texts)
        image_input, text_inputs = images.to(device), texts.to(device)

        # Forward pass
        if use_original_clip:
            logits_per_image, logits_per_text = model(image_input, text_inputs)
        else:
            logits_per_image, logits_per_text = inference_with_model(model, image_input, text_inputs)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

        if mode == "train":
            if use_grad_accum:
                loss = loss / accum_steps
                loss.backward()
                accumulation_step += 1
                
                if accumulation_step == accum_steps:
                    if device.type != "cpu":
                        convert_models_to_fp32(model)
                        optimizer.step()
                        clip.model.convert_weights(model)
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    accumulation_step = 0
                
                epoch_loss += loss.item() * accum_steps
            else:
                optimizer.zero_grad()
                loss.backward()
                if device.type != "cpu":
                    convert_models_to_fp32(model)
                    optimizer.step()
                    clip.model.convert_weights(model)
                else:
                    optimizer.step()
                epoch_loss += loss.item()
        else:
            epoch_loss += loss.item()

    # Handle remaining accumulated gradients
    if use_grad_accum and mode == "train" and 'accumulation_step' in locals() and accumulation_step > 0:
        if device.type != "cpu":
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        else:
            optimizer.step()
        optimizer.zero_grad()

    return epoch_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune CLIP on 4 datasets')
    parser.add_argument('--config', default='config_finetuning.yaml', 
                       help='Path to configuration file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    validate_common_config(config)
    setup_logging(config['logging']['log_level'])
    
    log_training_config(config, "Fine-tuning")
    
    device = setup_device(config)
    initialize_deterministic_mode(config['training']['seed'])

    batch_size = config['training']['batch_size']
    effective_batch_size = calculate_effective_batch_size(config)
    
    # Generate configuration name for tracking
    additional_info = ""
    if config['checkpoints']['checkpoint_to_load']:
        checkpoint_path = config['checkpoints']['checkpoint_to_load']
        if "train_OSM" in checkpoint_path:
            date_time = "_".join(checkpoint_path.split("/")[-1].split("_")[1:3])
            additional_info = f"after_pretrain_OSM_{date_time}_"
    
    conf_name = (f"{get_formatted_datetime()}_Task:FT_{additional_info}"
                f"OrigClip:{config['model']['use_original_clip']}_"
                f"Opt:{config['training']['optimizer']}_"
                f"AUG:{config['model']['use_augmentation']}_"
                f"DS:4_LR:{config['training']['learning_rate']}_"
                f"E:{config['training']['epochs']}_BS:{effective_batch_size}")
    
    # Setup checkpoint directory
    if config['checkpoints']['save_enabled']:
        output_dir = config['checkpoints']['output_dir']
        if config['gradient_accumulation']['enabled']:
            output_dir += "_gradient_accumulation"
        saving_models_path = os.path.join(output_dir, conf_name)
        os.makedirs(saving_models_path, exist_ok=True)
    
    if config['logging']['wandb_enabled']:
        initialize_wandb_logging(
            conf_name, 
            config['training']['learning_rate'], 
            config['training']['epochs'], 
            True, 
            conf_name
        )
    
    # Load model
    model_name = config['checkpoints']['checkpoint_to_load'] or config['model']['name']
    
    if config['model']['use_original_clip']:
        model, preprocess = clip.load(model_name, device=device)
    else:
        model, preprocess = _load_osm_clip(
            config['model']['use_augmentation'], 
            name=model_name, 
            device=device, 
            return_cls=True
        )
    
    optimizer = setup_optimizer(model, config)
    loss_img, loss_txt = torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()
    
    train_loader, val_loader, test_loader = load_datasets_as_single_one(
        config['datasets']['names'],
        batch_size,
        preprocess,
        config['augmentation']['aug_new_images'],
        config['augmentation']['num_aug_images'],
    )
    
    logging.info(f"Datasets loaded - Train: {len(train_loader.dataset)}, "
                f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Save initial model
    if config['checkpoints']['save_enabled']:
        save_checkpoint(saving_models_path, f"InitialModel_{conf_name}", model, 0)
    
    # Training loop
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    patience = config['training']['patience']
    
    for epoch in range(config['training']['epochs']):
        logging.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        
        train_loss = run_epoch(model, train_loader, optimizer, loss_img, loss_txt, device, config, "train")
        val_loss = run_epoch(model, val_loader, optimizer, loss_img, loss_txt, device, config, "eval")
        
        if config['logging']['wandb_enabled']:
            wandb.log({
                "train_epoch_loss": train_loss, 
                "val_epoch_loss": val_loss
            }, step=epoch)
        
        logging.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Check for improvement and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            
            if config['checkpoints']['save_enabled'] and config['checkpoints']['save_best_each_epoch']:
                save_checkpoint(saving_models_path, f"BestModel_{conf_name}", model, epoch)
        else:
            epochs_since_improvement += 1
        
        # Early stopping
        if epochs_since_improvement >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Final evaluation
    test_loss = run_epoch(model, test_loader, optimizer, loss_img, loss_txt, device, config, "eval")
    logging.info(f"Final Test Loss: {test_loss:.6f}")
    
    if config['logging']['wandb_enabled']:
        wandb.log({"total_test_loss": test_loss})
    
    # Save final model
    if config['checkpoints']['save_enabled']:
        save_checkpoint(saving_models_path, f"FinalBestModel_{conf_name}", model, None)
    
    if config['logging']['wandb_enabled']:
        wandb.finish()


if __name__ == "__main__":
    main()