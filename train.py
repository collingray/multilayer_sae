import torch
import tqdm
import gc

from aelib import ActivationsBuffer, ActivationsBufferConfig
from aelib.multilayer import AutoEncoderMultiLayerConfig, AutoEncoderMultiLayerTrainer, \
    AutoEncoderMultiLayerTrainerConfig

model_name = "EleutherAI/pythia-70m"
dataset_name = "wikitext"
dataset_config = "wikitext-103-v1"
dataset_split = "train"

seed = 42
torch.manual_seed(seed)

layers = list(range(6))

# Buffer to easily generate/store activations from the model, automatically refills the buffer when it drops below a certain size,
# and then shuffles it to avoid activations from the same sequence being used together
buffer_cfg = ActivationsBufferConfig(
    model_name=model_name,
    layers=layers,
    dataset_name=dataset_name,
    dataset_config=dataset_config,
    dataset_split=dataset_split,
    act_site="hook_mlp_out",
    device="cuda",
    buffer_device="cpu",
    buffer_size=2 ** 21,
    min_capacity=2 ** 17,
    model_batch_size=256,
    samples_per_seq=1024,
    max_seq_length=2048,
    seed=seed
)

buffer = ActivationsBuffer(buffer_cfg)

total_activations = int(5e8)
batch_size = 8192
expansion_factor = 12
n_dim = 512
lambda_reg = 4e-2

autoencoder_cfg = AutoEncoderMultiLayerConfig(
    n_dim=n_dim,
    m_dim=n_dim * expansion_factor,
    lambda_reg=lambda_reg,
    record_data=True,
    save_dir="./weights",
    seed=seed,
)

autoencoder_trainer_cfg = AutoEncoderMultiLayerTrainerConfig(
    lr=4e-3,
    beta1=0,
    beta2=0.99,
    total_steps=total_activations // batch_size,
    warmup_percent=0.05,
    steps_per_report=128,
    steps_per_resample=8e7 // batch_size,
    num_resamples=4,
    wb_project="multilayer_sae",
    wb_entity="collingray",
    wb_name="gpt2_wikitext_16x_v2",
    wb_group="gpt2_wikitext",
    wb_config=autoencoder_cfg.__dict__,
)

autoencoder_trainer = AutoEncoderMultiLayerTrainer(autoencoder_cfg, autoencoder_trainer_cfg)

try:
    for _ in tqdm(range(total_activations // batch_size)):
        acts = buffer.next(batch=batch_size).to(autoencoder_cfg.device, dtype=autoencoder_cfg.dtype)
        autoencoder_trainer.train_on(acts, buffer)
finally:
    autoencoder_trainer.finish()