import fire

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader



from eb_jepa.training_utils import (
    load_config,
    setup_device,
    setup_seed,
)
from eb_jepa.datasets.robosuite import VideoRobosuiteDataset

from eb_jepa.architectures import RNNEncoder, ResUNet, Projector, StateOnlyPredictor, Transformer
from eb_jepa.logging import get_logger
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VCLoss


logger = get_logger(__name__)


def run(
    fname: str = "examples/recurrent_enc/cfgs/default.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    
    # Load config
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Setup using shared utilities
    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    # # Create experiment directory using unified structure (if not provided)
    # if folder is None:
    #     if cfg.meta.get("model_folder"):
    #         exp_dir = Path(cfg.meta.model_folder)
    #         folder_name = exp_dir.name
    #         exp_name = folder_name.rsplit("_seed", 1)[0]
    #     else:
    #         sweep_name = get_default_dev_name()
    #         exp_name = get_exp_name("image_jepa", cfg)
    #         exp_dir = get_unified_experiment_dir(
    #             example_name="image_jepa",
    #             sweep_name=sweep_name,
    #             exp_name=exp_name,
    #             seed=cfg.meta.seed,
    #         )
    # else:
    #     exp_dir = Path(folder)
    #     exp_dir.mkdir(parents=True, exist_ok=True)
    #     # Extract exp_name from folder name by removing _seed{seed} suffix
    #     folder_name = exp_dir.name  # e.g., "resnet_vicreg_seed1"
    #     exp_name = folder_name.rsplit("_seed", 1)[0]  # e.g., "resnet_vicreg"

    # wandb_run = setup_wandb(
    #     project="eb_jepa",
    #     config={"example": "image_jepa", **OmegaConf.to_container(cfg, resolve=True)},
    #     run_dir=exp_dir,
    #     run_name=exp_name,
    #     tags=["image_jepa", f"seed_{cfg.meta.seed}"],
    #     group=cfg.logging.get("wandb_group"),
    #     enabled=cfg.logging.log_wandb,
    #     sweep_id=cfg.logging.get("wandb_sweep_id"),
    # )

    logger.info("Loading robosuite dataset...")
    train_dataset = VideoRobosuiteDataset(cfg.data.data_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid small batches that cause BatchNorm issues
    )
    
    #transform = get_train_transforms()

    # Use EBJEPA_DSETS environment variable if set, otherwise fall back to config
    #data_dir = os.environ.get("EBJEPA_DSETS")
    #logger.info(f"Using data directory: {data_dir}")

    # base_train_dataset = CIFAR10(
    #     root=data_dir, train=True, download=True, transform=None
    # )

    # train_dataset = ImageDataset(base_train_dataset, transform, num_crops=2)

    # val_dataset = CIFAR10(
    #     root=data_dir, train=False, download=True, transform=get_val_transforms()
    # )

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=cfg.data.batch_size,
    #     shuffle=True,
    #     num_workers=cfg.data.num_workers,
    #     pin_memory=True,
    #     drop_last=True,  # Avoid small batches that cause BatchNorm issues
    # )

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=cfg.data.batch_size,
    #     shuffle=False,
    #     num_workers=cfg.data.num_workers,
    #     pin_memory=True,
    # )

    # log_data_info(
    #     "CIFAR-10",
    #     len(train_loader),
    #     cfg.data.batch_size,
    #     train_samples=len(train_dataset),
    #     val_samples=len(val_dataset),
    # )

    # Initialize model
    logger.info("Initializing model...")
    encoder = RNNEncoder(input_size=cfg.model.dobs, hidden_size=cfg.model.henc)
    predictor_model = Transformer(dim=cfg.model.henc, context_length=2)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)

    projector = None #Projector(f"{cfg.model.dstc}-{cfg.model.dstc*4}-{cfg.model.dstc*4}")

    regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)
    ploss = SquareLossSeq(projector)
    jepa = JEPA(encoder, None, predictor, regularizer, ploss).to(device)

    optimizer = AdamW(
        jepa.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.get("weight_decay", 1e-6),
    )
    
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        video = batch["video"]
        robot_state = batch['robot_state']
        robot_action = batch['robot_action']

        video = video.permute(0,2,1,3,4).flatten(2)
        x = torch.cat([video, robot_state, robot_action], axis=2)[:, :, :100]
        print(x.shape)
        
        optimizer.zero_grad()
        _, (jepa_loss, regl, _, regldict, pl) = jepa.unroll(
            x,
            actions=None,
            nsteps=cfg.model.steps,
            unroll_mode="parallel",
            compute_loss=True,
            return_all_steps=False,
        )
        jepa_loss.backward()
        optimizer.step()


if __name__ == "__main__":
    fire.Fire(run)
