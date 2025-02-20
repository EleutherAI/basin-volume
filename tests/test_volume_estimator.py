import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tyche import VolumeConfig, VolumeEstimator

def test_volume_estimator_and_model_parameters():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m", device_map=device)
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 0
    
    dataset = load_dataset("EleutherAI/lambada_openai", name="en", split="test", trust_remote_code=True)
    
    cfg_non_implicit = VolumeConfig(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        text_key="text",
        n_samples=40,
        cutoff=1e-2,
        max_seq_len=2048,
        val_size=10,
        chunking=False,
        implicit_vectors=False
    )
    estimator_non_imp = VolumeEstimator.from_config(cfg_non_implicit)
    result_non_imp = estimator_non_imp.run()
    
    cfg_implicit = VolumeConfig(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        text_key="text",
        n_samples=40,
        cutoff=1e-2,
        max_seq_len=2048,
        val_size=10,
        chunking=False,
        implicit_vectors=True
    )
    estimator_imp = VolumeEstimator.from_config(cfg_implicit)
    result_imp = estimator_imp.run()
    
    std_non_imp = torch.std(result_non_imp.estimates)
    diff = torch.abs(result_non_imp.estimates.mean() - result_imp.estimates.mean())
    assert torch.all(diff < 5 * std_non_imp/torch.sqrt(torch.tensor(len(result_non_imp.estimates)))), "Implicit and non-implicit estimates differ too much"
    print(f"normalized diff: {diff/std_non_imp/torch.sqrt(torch.tensor(len(result_non_imp.estimates)))}")
    
    model2 = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m").to(device)
    vec1 = torch.nn.utils.parameters_to_vector(model.parameters())
    vec2 = torch.nn.utils.parameters_to_vector(model2.parameters())
    assert torch.allclose(vec1, vec2, rtol=1e-3, atol=1e-3), "Model parameters do not match between copies" 