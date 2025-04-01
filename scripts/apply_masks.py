import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from dataset_processor import CHAT_TEMPLATES
from utils import ArgumentParserPlus

@dataclass
class FlatArguments:
    """
    Full arguments class for all fine-tuning jobs.
    """
    base_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path or name to the base model."
            )
        },
    )
    mask_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path or name to the model with scores."
            )
        },
    )
    masks_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The saved masks."
            )
        },
    )
    output_dir: str = field(
        default="output/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    # Add for customer masks
    seed: int = field(default=42, metadata={"help": "Random seed for random mask initialization."})
    subnet_mode: Optional[str] = field(
        default=None,
    )
    sparsity_attn: Optional[float] = field(
        default=None,
    )
    sparsity_mlp: Optional[float] = field(
        default=None,
    )
    masked_layers: Optional[str] = field(
        default=None,
    )
    random_masks: bool = field(
        default=False,
    )
    save_masks: bool = field(
        default=False,
    )
    only_save_masks: bool = field(
        default=False,
    )
    use_l1_mask: bool = field(
        default=False,
        metadata={"help": "Whether to use L1 norm for mask generation"},
    )

def percentile(t, q):
    k = 1 + round(float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()

def generate_l1_masks(model, sparsity_attn, sparsity_mlp, masked_layers, subnet_mode):
    """
    Generate masks based on L1 norm of parameters for specific layers.
    
    Args:
        model: The model to generate masks for
        sparsity_attn: Target sparsity ratio for attention layers
        sparsity_mlp: Target sparsity ratio for MLP layers
        masked_layers: List of layer indices to apply masks
        subnet_mode: Mode for masking ('attn', 'mlp', or 'both')
    
    Returns:
        dict: Dictionary containing masks for each parameter
    """
    masks = {}
    vocab_related_layers = ['wte', 'embed', 'lm_head', 'norm']
    
    for name, param in model.named_parameters():
        # Skip vocab-related layers
        if any(vocab_str in name for vocab_str in vocab_related_layers):
            continue

        # Get layer index and check if it should be masked
        try:
            layer_index = int(name.split(".")[2])
            if layer_index not in masked_layers:
                continue
        except (IndexError, ValueError):
            continue

        # Skip based on subnet_mode
        if subnet_mode == "attn" and "mlp" in name:
            continue
        if subnet_mode == "mlp" and "self_attn" in name:
            continue

        # Calculate L1 norm
        shape = param.data.shape
        if "self_attn" in name:
            if subnet_mode in ["attn", "both"]:
                sparsity = sparsity_attn
                score = param.abs().mean(dim=[1, 2, 3]) if len(shape) == 4 else param.abs().mean(dim=1)
            else:
                continue
        elif "mlp" in name:
            if subnet_mode in ["mlp", "both"]:
                sparsity = sparsity_mlp
                score = param.abs().mean(dim=[1, 2, 3]) if len(shape) == 4 else param.abs().mean(dim=1)
            else:
                continue
        else:
            continue

        # Determine number of elements to prune
        n_elements = len(score)
        n_pruned = min(int((1 - sparsity) * n_elements), n_elements - 1)  # 修改这里，使用(1-sparsity)作为剪枝比例

        # Get pruning indices (smallest L1 norms)
        sorted_indices = score.sort()[1]
        pruned_indices = sorted_indices[:n_pruned].cpu()
        
        # Create mask (1 for kept, 0 for pruned)
        mask = torch.ones_like(score)
        mask[pruned_indices] = 0

        # Expand mask to match parameter shape if needed
        mask = mask.view(-1, 1).expand_as(param)

        module_name = name[:-7] if name.endswith(".weight") else name
        masks[module_name] = mask
        print(f"Generated L1 mask for {module_name}")
        print(f"  - Layer: {layer_index}")
        print(f"  - Type: {'attention' if 'self_attn' in name else 'mlp'}")
        print(f"  - Target sparsity: {sparsity:.6f}")
        print(f"  - Actual sparsity: {1 - torch.mean(mask):.6f}")
    
    return masks

def main(args: FlatArguments):
    if args.seed is not None:
        set_seed(args.seed)

    # Load fft model and tokenizer
    config_base = AutoConfig.from_pretrained(
        args.base_model_name_or_path,
    )

    model_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        config=config_base,
    )

    base_params = model_base.state_dict()
    
    masks = {}
    vocab_related_layers = ['wte', 'embed', 'lm_head', 'norm']
    if args.use_l1_mask:
        assert args.masked_layers, "Need the index of layers to generate L1 masks."
        masked_layers = [int(x) for x in args.masked_layers.split()]
        assert args.subnet_mode, "Need the subnet_mode to define where to add L1 masks."
        
        if args.subnet_mode == "attn" or args.subnet_mode == "both":
            assert args.sparsity_attn, "Need the sparsity_attn to generate L1 masks."
        if args.subnet_mode == "mlp" or args.subnet_mode == "both":
            assert args.sparsity_mlp, "Need the sparsity_mlp to generate L1 masks."
            
        print(f"Generating L1 masks for layers {masked_layers} in {args.subnet_mode} mode...")
        masks = generate_l1_masks(
            model_base,
            args.sparsity_attn,
            args.sparsity_mlp,
            masked_layers,
            args.subnet_mode
        )
    elif args.random_masks:
        assert args.masked_layers, "Need the index of layers to generate random masks."
        masked_layers = [int(x) for x in args.masked_layers.split()] if args.masked_layers else args.masked_layers
        assert args.subnet_mode, "Need the subnet_mode to define where to add random masks."
        if args.subnet_mode == "attn" or args.subnet_mode == "both":
           assert args.sparsity_attn, "Need the sparsity_attn to generate random masks."
        elif args.subnet_mode == "mlp" or args.subnet_mode == "both":
            assert args.sparsity_mlp, "Need the sparsity_mlp to generate random masks."

        for name, param in model_base.named_parameters():
            if any(vocab_str in name for vocab_str in vocab_related_layers):
                continue
            if ((args.subnet_mode == "attn") and ("mlp" in name)) or ((args.subnet_mode == "mlp") and ("self_attn" in name)):
                continue
            layer_index = int(name.split(".")[2])
            sparsity = args.sparsity_attn if 'self_attn' in name else args.sparsity_mlp
            if layer_index in masked_layers:
                mask = (torch.rand(param.size(), device=param.device) < sparsity).to(dtype=param.dtype)
                print(f"Ratio of 1s: {torch.mean(mask):.8f}, Ratio of 0s: {1 - torch.mean(mask):.8f} (target sparsity: {sparsity:.8f})")
                module_name = name[:-7]
                masks[module_name] = mask
    else:
        assert (not args.random_masks) and (not args.use_l1_mask)
        assert args.mask_model_name_or_path or args.masks_file, "There's no masks to apply."

        if args.mask_model_name_or_path:
            config_mask = AutoConfig.from_pretrained(args.mask_model_name_or_path)

            model_mask = AutoModelForCausalLM.from_pretrained(
                args.mask_model_name_or_path,
                config=config_mask,
            )
        
            for name, param in model_mask.named_parameters():
                if name not in base_params:
                    assert name.endswith(".scores"), f"Found an unexpected parameter: {name}"
                    scores = param.abs()
                    sparsity = config_mask.sparsity_attn if 'self_attn' in name else config_mask.sparsity_mlp

                    mask = scores.clone()
                    _, idx = scores.flatten().sort()
                    j = int((1 - sparsity) * scores.numel())
                    # flat_mask and mask access the same memory.
                    flat_mask = mask.flatten()
                    flat_mask[idx[:j]] = 0
                    flat_mask[idx[j:]] = 1

                    module_name = name[:-7]
                    masks[module_name] = mask
                    
                    continue
                
                if any(vocab_str in name for vocab_str in vocab_related_layers):
                    module_size = param.data.shape[0]
                    checkpoint_size = base_params[name].shape[0]
                    if module_size != checkpoint_size:
                        print(f"Note: Vocab size mismatch in {name}: model={module_size}, checkpoint={checkpoint_size}")
                    continue

                assert torch.equal(param.data, base_params[name]), f"Weights mismatch in parameter {name}."
        
        else:
            masks = torch.load(args.masks_file, map_location="cpu")

    assert masks, f"There're no masks."

    if args.save_masks:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(masks, os.path.join(args.output_dir, "masks.pt"))
    if args.only_save_masks:
        return
    
    have_scores = False
    for name, module in model_base.named_modules():
        if name in masks:
            have_scores = True
            masks[name] = masks[name].to(module.weight.device)
            module.weight.data.mul_(masks[name])
            print(f"Applied mask on {name} with sparsity {1 - torch.mean(masks[name]):.8f}.")
    assert have_scores == True, f"There're no masks."

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path
    )

    tokenizer.save_pretrained(args.output_dir)
    model_base.save_pretrained(args.output_dir,state_dict=model_base.state_dict(),safe_serialization=False)

if __name__ == "__main__":
    parser = ArgumentParserPlus((FlatArguments))
    args = parser.parse()
    main(args)