import torch
import os
import clip
import argparse
from tqdm import tqdm
from create_model import _load_osm_clip
from utils.dataset_utils import load_osm_dataset
from utils.general_utils import (
    convert_models_to_fp32,
    get_formatted_datetime,
    initialize_deterministic_mode,
    save_checkpoint,
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
import logging
from utils.logging_utils import initialize_wandb_logging
from dataset import _pluck_patches, custom_collate_fn


def validate_osm_config(config):
    """Validate OSM-specific configuration parameters."""
    # First run common validation
    validate_common_config(config)
    
    # OSM-specific validations
    patch_size = config['model']['patch_size']
    image_size = config['model']['image_size']
    
    if image_size % patch_size != 0:
        raise ValueError("Image size must be divisible by patch size")
    
    if config['model']['use_original_clip']:
        raise NotImplementedError("Original CLIP model not supported for training on OSM")
    
    # Check dataset path exists
    if not os.path.exists(config['dataset']['path']):
        raise FileNotFoundError(f"Dataset path does not exist: {config['dataset']['path']}")


def run_epoch(model, dataloader, optimizer, loss_img, loss_txt, device, config, mode="train"):
    """Run one epoch of training or validation."""
    epoch_loss = 0
    model.train() if mode == "train" else model.eval()
    
    use_grad_accum = config['gradient_accumulation']['enabled']
    accum_steps = config['gradient_accumulation']['steps']
    patch_size = config['model']['patch_size']
    image_size = config['model']['image_size']
    
    if mode == "train" and use_grad_accum:
        optimizer.zero_grad()
        accumulation_step = 0

    for batch in tqdm(dataloader, desc=f"{mode.capitalize()}"):
        images, tags_patches_batch, tokenized_tags = batch
        images = images.to(device)
        tokenized_tags = tokenized_tags.to(device)
        
        # Forward pass
        image_features, text_features = model(images, tokenized_tags)
        plucked_patch_features = _pluck_patches(
            image_features, tags_patches_batch, image_size, patch_size
        )

        target = torch.arange(0, len(tags_patches_batch), dtype=torch.long, device=device)
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * plucked_patch_features @ text_features.T
        logits_per_text = logits_per_image.T

        # Calculate loss
        loss = (loss_img(logits_per_image, target) + loss_txt(logits_per_text, target)) / 2
        
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

    # Handle remaining gradients in gradient accumulation mode
    if mode == "train" and use_grad_accum and 'accumulation_step' in locals() and accumulation_step > 0:
        if device.type != "cpu":
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        else:
            optimizer.step()
        optimizer.zero_grad()

    return epoch_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train CLIP on OSM dataset')
    parser.add_argument('--config', default='config_train_osm.yaml', 
                       help='Path to configuration file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    validate_osm_config(config)
    setup_logging(config['logging']['log_level'])
    
    log_training_config(config, "OSM Training")
    
    device = setup_device(config)
    initialize_deterministic_mode(config['training']['seed'])
    
    batch_size = config['training']['batch_size']
    effective_batch_size = calculate_effective_batch_size(config)
    
    # Generate run name and paths
    patch_size = config['model']['patch_size']
    accumulation_suffix = f"_GA{config['gradient_accumulation']['steps']}" if config['gradient_accumulation']['enabled'] else ""
    task_name = f"unified_osm_training_{get_formatted_datetime()}"
    run_name = (f"OrigClip:{config['model']['use_original_clip']}_"
               f"Opt:{config['training']['optimizer']}_"
               f"AUG:{config['model']['use_augmentation']}_DS:OSM{accumulation_suffix}")
    
    conf_name = (f"{get_formatted_datetime()}_Task:{task_name}_{run_name}_"
                f"LR:{config['training']['learning_rate']}_E:{config['training']['epochs']}_"
                f"BS:{effective_batch_size}")
    
    # Setup checkpoint directory
    if config['checkpoints']['save_enabled']:
        saving_models_path = os.path.join(config['checkpoints']['output_dir'], conf_name)
        os.makedirs(saving_models_path, exist_ok=True)
    
    if config['logging']['wandb_enabled']:
        initialize_wandb_logging(
            f"ViT-B/{patch_size}", 
            config['training']['learning_rate'], 
            config['training']['epochs'], 
            True, 
            conf_name
        )
    
    # Load model
    model, preprocess = _load_osm_clip(
        config['model']['use_augmentation'],
        name=f"ViT-B/{patch_size}",
        device=device,
        return_cls=config['model']['return_cls_for_osm'],
    )
    
    optimizer = setup_optimizer(model, config)
    loss_img, loss_txt = torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()
    
    # Load dataset
    mask_path = config['dataset']['mask_path']
    if config['gradient_accumulation']['enabled'] and mask_path is None:
        mask_path = "masks/32_f1"
    
    train_loader, val_loader, test_loader = load_osm_dataset(
        config['dataset']['path'],
        batch_size,
        config['model']['image_size'],
        config['dataset']['train_percentage'],
        config['dataset']['val_percentage'],
        preprocess,
        patch_size,
        custom_collate_fn,
        mask_path,
        mask_path,
    )
    
    train_size = len(train_loader.dataset.indices)
    val_size = len(val_loader.dataset.indices)
    test_size = len(test_loader.dataset.indices)
    
    logging.info(f"Dataset loaded - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Save initial model
    if config['checkpoints']['save_enabled']:
        save_checkpoint(saving_models_path, f"InitialModel_{conf_name}", model, 0)
    
    # Training loop
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    patience = config['training']['patience']
    
    for epoch in range(config['training']['epochs']):
        logging.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        
        # Training
        train_loss = run_epoch(model, train_loader, optimizer, loss_img, loss_txt, device, config, "train")
        
        # Validation
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, optimizer, loss_img, loss_txt, device, config, "eval")

        if config['logging']['wandb_enabled']:
            wandb.log({
                "train_epoch_loss": train_loss,
                "val_epoch_loss": val_loss,
                "learning_rate": config['training']['learning_rate'],
                "epoch": epoch
            }, step=epoch)

        logging.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            
            if config['checkpoints']['save_enabled'] and config['checkpoints']['save_best_each_epoch']:
                save_checkpoint(saving_models_path, f"BestModel_{conf_name}", model, epoch)
            
            logging.info(f"New best validation loss: {best_val_loss:.6f}")
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Final evaluation on test set
    logging.info("Running final evaluation on test set...")
    with torch.no_grad():
        test_loss = run_epoch(model, test_loader, optimizer, loss_img, loss_txt, device, config, "eval")
        
    logging.info(f"Final Test Loss: {test_loss:.6f}")

    if config['logging']['wandb_enabled']:
        wandb.log({"total_test_loss": test_loss})

    # Save final model
    if config['checkpoints']['save_enabled']:
        save_checkpoint(saving_models_path, f"FinalBestModel_{conf_name}", model, None)
        logging.info(f"Final model saved to: {saving_models_path}")

    logging.info("Training completed successfully!")
    
    if config['logging']['wandb_enabled']:
        wandb.finish()


if __name__ == "__main__":
    main()