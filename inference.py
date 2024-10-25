from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class LLaMa:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints found"
            chk_path = checkpoints[0]
            print(f"Loading Checkpoint {chk_path}")
            checkpoints = torch.load(chk_path, map_location=device)  # Map to correct device
            print(f"Loaded Checkpoint in {(time.time() - prev_time):.2f}s")
            prev_time = time.time()

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        
        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)  # Use float32 for compatibility
        else:
            torch.set_default_tensor_type(torch.FloatTensor)  # Use float32 on CPU

        model = Transformer(model_args).to(device)

        if load_model:
            if "rope.freqs" in checkpoints:  # Check if the key exists before deleting
                del checkpoints["rope.freqs"]
            model.load_state_dict(checkpoints, strict=True)
            print(f"Loaded state dict in {(time.time() - prev_time):.2f}s")

        return LLaMa(model, tokenizer, model_args)


if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False  # Set this to True if you want to allow GPU usage
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"
    
    try:
        model = LLaMa.build(
            checkpoints_dir='llama-2-7b',  # Ensure this directory exists
            tokenizer_path="tokenizer.model",  # Ensure this file exists
            load_model=True,  # Should be boolean, not a string
            max_seq_len=1024,
            max_batch_size=3,
            device=device
        )
        print("All Ok")
    except Exception as e:
        print(f"Error occurred: {e}")