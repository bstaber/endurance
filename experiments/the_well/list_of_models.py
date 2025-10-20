"""List the number of trainable parameters for all models and datasets from The Well."""

import warnings

from the_well.benchmark.models import FNO, TFNO, UNetClassic, UNetConvNext

# Silence the irrelevant warnings from pydantic/timm
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\._internal\._generate_schema")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\.models\.layers")

DATASETS = [
    "acoustic_scattering_maze",
    "active_matter",
    "convective_envelope_rsg",
    "gray_scott_reaction_diffusion",
    "helmholtz_staircase",
    "MHD_64",
    "planetswe",
    "post_neutron_star_merger",
    "rayleigh_benard",
    "rayleigh_taylor_instability",
    "shear_flow",
    "supernova_explosion_64",
    "turbulence_gravity_cooling",
    "turbulent_radiative_layer_2D",
    "viscoelastic_instability",
]

MODELS = {
    "FNO": FNO,
    "TFNO": TFNO,
    "UNetClassic": UNetClassic,
    "UNetConvNext": UNetConvNext,
}


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print(f"{'Model':15s} | {'Dataset':35s} | {'#Trainable (M)':>15s} | {'#Total (M)':>12s}")
    print("-" * 90)

    for model_name, cls in MODELS.items():
        for dataset in DATASETS:
            repo = f"polymathic-ai/{model_name}-{dataset}"
            try:
                model = cls.from_pretrained(repo)
                total, trainable = count_params(model)
                print(f"{model_name:15s} | {dataset:35s} | {trainable/1e6:15.2f} | {total/1e6:12.2f}")
            except Exception as e:
                # Skip missing models silently or print short error
                print(f"{model_name:15s} | {dataset:35s} | {'--':>15s} | {'--':>12s}  ({type(e).__name__})")

    print("-" * 90)


if __name__ == "__main__":
    main()
