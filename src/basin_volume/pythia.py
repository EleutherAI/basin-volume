import json
from huggingface_hub import list_repo_refs
import torch
import jax
from transformers import AutoModelForCausalLM

def get_pythia_checkpoint_steps(model_name="14m"):
    branches = list_repo_refs(f"EleutherAI/pythia-{model_name}").branches
    branch_names = [b.name for b in branches]
    branch_names = [b for b in branch_names if b.startswith("step")]
    checkpoint_steps = [int(b.split("step")[1]) for b in branch_names]
    checkpoint_steps = sorted(checkpoint_steps)
    return checkpoint_steps

def load_pythia_checkpoint(step, model_name="14m"):
    model_chkpt = AutoModelForCausalLM.from_pretrained(f"EleutherAI/pythia-{model_name}", revision=f"step{step}").to("cuda")
    return model_chkpt

def load_pythia_checkpoint_states(step, model_name="14m"):
    """Load the checkpoint states from disk for a given step."""
    with open(f"/mnt/hdd-0/tiny-pythia/ckpts/pythia-{model_name}/global_step{step}/mp_rank_00_model_states.pt", "rb") as f:
        return torch.load(f)
    
def load_pythia_val_data(tokenizer, n_seqs=10):
    """Load the validation data from disk."""
    with open("/mnt/ssd-1/adam/basin-volume/data/pile_val.jsonl", "r") as f:
        text_val = [json.loads(line)['text'] for line in f]
    text_val = text_val[:n_seqs]
    X_val_t = tokenizer(text_val, return_tensors="pt", padding=True, truncation=True, max_length=1024)['input_ids'].to("cuda")

    return X_val_t

def match_params_to_flat(model_params, flat_params):
    """
    Match model parameters to their positions in the flattened optimizer state.

    Args:
        model_params: Iterator of (name, param) tuples from model.named_parameters()
        flat_params: Tensor containing flattened parameters

    Returns:
        List of (name, start_idx, numel) tuples
    """
    matches = []
    current_idx = 0

    # Convert flat_params to CPU for comparison
    flat_params_cpu = flat_params.cpu()

    while current_idx < len(flat_params):
        best_match = None
        best_diff = float('inf')
        # print(f"current_idx: {current_idx}")
        # print(f"len(flat_params): {len(flat_params)}")

        # print(f"# of params: {len(model_params)}")

        # Try each remaining parameter
        for name, param in model_params.items():
            # print(f"matching {name}")
            # Skip if we've already matched this param
            if any(name == m[0] for m in matches):
                # print(f"parameter {name} already matched")
                continue

            numel = param.numel()
            if current_idx + numel > len(flat_params):
                # print(f"parameter {name} is too big for group {group_idx} at {current_idx}")
                continue

            # Compare the flattened parameter with the slice of flat_params
            flat_slice = flat_params_cpu[current_idx:current_idx + numel]
            diff = torch.abs(param.detach().cpu().flatten() - flat_slice).mean().item()

            if diff < best_diff:
                best_diff = diff
                best_match = (name, current_idx, numel, diff)

        if best_match is None:
            print(f"No match found for group at {current_idx}")
            break

        matches.append(best_match)
        current_idx += best_match[2]

        # Print progress
        # print(f"Matched {best_match[0]}: {best_match[2]} params starting at {best_match[1]} (diff: {best_match[3]:.2e})")
        #print(f"index now at {current_idx} of {len(flat_params)}")

    return matches

def build_pythia_adam_vectors(model, states):
    """Extract and reconstruct ADAM states from checkpoint states."""
    all_matches = []
    optstates = states['optimizer']['optimizer_state_dict']['state']
    
    # Run the matching for each flat group
    for group_idx, flat_group in enumerate(states['optimizer']['fp32_groups_flat']):
        print(f"Matching parameters in group {group_idx}")
        matches = match_params_to_flat(dict(model.named_parameters()), flat_group)
        all_matches.append(matches)

    # Create dictionaries to store the matched states
    param_exp_avg = {}
    param_exp_avg_sq = {}
    param_fp32 = {}

    # Iterate through each group and its matches
    for group_idx, matches in enumerate(all_matches):
        opt_state = optstates[group_idx]
        exp_avg = opt_state['exp_avg']
        exp_avg_sq = opt_state['exp_avg_sq']
        fp32_params = states['optimizer']['fp32_groups_flat'][group_idx]

        for name, start_idx, numel, diff in matches:
            param_exp_avg[name] = exp_avg[start_idx:start_idx + numel]
            param_exp_avg_sq[name] = exp_avg_sq[start_idx:start_idx + numel]
            param_fp32[name] = fp32_params[start_idx:start_idx + numel]

    # Reconstruct vectors in model parameter order
    exp_avg_vec = []
    exp_avg_sq_vec = []
    fp32_vec = []

    for name, param in model.named_parameters():
        exp_avg_vec.append(param_exp_avg[name].flatten())
        exp_avg_sq_vec.append(param_exp_avg_sq[name].flatten())
        fp32_vec.append(param_fp32[name].flatten())

    # Concatenate into single tensors
    exp_avg_reconstructed = torch.cat(exp_avg_vec)
    exp_avg_sq_reconstructed = torch.cat(exp_avg_sq_vec)
    fp32_reconstructed = torch.cat(fp32_vec)

    # Convert to JAX arrays
    adam1 = jax.dlpack.from_dlpack(exp_avg_reconstructed.cuda())
    adam2 = jax.dlpack.from_dlpack(exp_avg_sq_reconstructed.cuda())

    # assert fp32_reconstructed ~= model.parameters()
    # print(torch.norm(fp32_reconstructed - torch.nn.utils.parameters_to_vector(model.parameters())))
    # print(torch.norm(fp32_reconstructed))
    assert torch.allclose(fp32_reconstructed, 
                          torch.nn.utils.parameters_to_vector(model.parameters()), 
                          rtol=1e-3, 
                          atol=1e-3)
    
    return adam1, adam2
