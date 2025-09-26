import wandb

def initialize_wandb_logging(MODEL_NAME, LEARNING_RATE, EPOCHS, WANDB_LOGGING, CONF_NAME):
    if WANDB_LOGGING:
        wandb.init(
        # set the wandb project where this run will be logged
        project="OpenStreetCLIP",
        entity="melganilab",
        name=CONF_NAME,
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": MODEL_NAME,
        "dataset": "SICD_UCM_NPWU_SIDNEY",
        "epochs": EPOCHS,
        }
    )