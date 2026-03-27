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

from eb_jepa.architectures import RNNEncoder, ResUNet, ResNet5, FeedForward, Projector, Transformer
from eb_jepa.logging import get_logger
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VCLoss
from eb_jepa.datasets.utils import init_data


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
    loader, val_loader, data_config = init_data(
        env_name=cfg.data.env_name, cfg_data=dict(cfg.data)
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

    video_encoder = ResNet5(in_d=2, h_d=cfg.model.v_enc.hdim, out_d=cfg.model.v_enc.odim, avg_pool=True)
    action_encoder = FeedForward(dim=2, hidden_dim=cfg.model.a_enc.hdim, output_dim=cfg.model.a_enc.odim)
    input_encoder = RNNEncoder(input_size=cfg.model.dobs, hidden_size=cfg.model.henc)
    
    #predictor_model = Transformer(dim=cfg.model.henc, input_channels=2*cfg.data.num_channels, context_length=2)
    #predictor = StateOnlyPredictor(predictor_model, context_length=2)

    #projector = None #Projector(f"{cfg.model.dstc}-{cfg.model.dstc*4}-{cfg.model.dstc*4}")

    # regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)
    # ploss = SquareLossSeq(projector)
    # jepa = JEPA(encoder, None, predictor, regularizer, ploss).to(device)

    # optimizer = AdamW(
    #     jepa.parameters(),
    #     lr=cfg.optim.lr,
    #     weight_decay=cfg.optim.get("weight_decay", 1e-6),
    # )
    
    for (x, a, loc, _, _) in loader:
        x_encoded = video_encoder(x)
        a_encoded = action_encoder(a)
        print(x_encoded.shape)
        print(a_encoded.shape)
        # a_encoded = action_encoder(a.permute(0,2,1))
        # observations = torch.cat([x_encoded, a_encoded])
        
        # print('Observations shape', observations.shape)
        
        # optimizer.zero_grad()
        # _, (jepa_loss, regl, _, regldict, pl) = jepa.unroll(
        #     x,
        #     actions=None,
        #     nsteps=cfg.model.steps,
        #     unroll_mode="parallel",
        #     compute_loss=True,
        #     return_all_steps=False,
        # )
        # jepa_loss.backward()
        # optimizer.step()


if __name__ == "__main__":
    fire.Fire(run)
