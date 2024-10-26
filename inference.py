from typing import Optional, List
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

    # def text_completion(self, prompt: str, candidates: List[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
    #     if max_gen_len is None:
    #         max_gen_len = self.args.max_seq_len
    #     # Convert each prompt into tokens
    #     prompt_tokens = [self.tokenizer.encode(prompt,out_type=int,add_bos=True,add_eos=True) for prompt in prompts]
    #     batch_size = len(prompt_tokens)
    #     assert batch_size <= self.args.max_batch_size, f"Batch size {batch_size} exceeds the maximum batch size {self.args.max_batch_size}"
    #     # Make sure the prompt length is not larger than the maximum sequence length
    #     max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
    #     total_len=min(self.args.max_seq_len,max_gen_len+max_prompt_len)
        
    #     # Create the list that will contain the generated tokens,along the intial tokens
    #     pad_id=self.tokenizer.pad_id()
    #     tokens=torch.full((batch_size,total_len),pad_id,dtype=torch.long,device=device)
    #     for k,t in enumerate(prompt_tokens):
    #         tokens[k,:len(t)]=torch.tensor(t,dtype=torch.long,device=device)
    #     eos_reached=torch.tensor([False]*batch_size).to(device)
    #     prompt_tokens_mask=tokens!=pad_id
    #     for cur_pos in tqdm(range(1,total_len),desc="Generating Tokens"):
    #         with torch.no_grad():
    #             logits=self.model.forward(tokens[:,cur_pos-1:cur_pos],cur_pos)
    #         # Apply temperature
    #         if temperature>0:
    #                 probs=torch.softmax(logits[:,-1]/temperature,dim=-1)
    #                 next_token=self._sample_top_p(probs,top_p)
    #         else:
    #             next_token=torch.argmax(logits[:,-1],dim=-1)
            
    #         next_token=next_token.reshape(-1)

    #         next_token=torch.where(prompt_tokens_mask[:cur_pos],tokens[:cur_pos],tokens[:cur_pos,],next_token)
    #         tokens[:,cur_pos]=next_token
    #         eos_reached|=(~prompt_tokens_mask[:,cur_pos])&(next_token==self.tokenizer.eos_id())
    #         if all(eos_reached):
    #             break

    #     out_tokens=[]
    #     out_text=[]
    #     for prompt_index,current_prompt_tokens in enumerate(tokens):
    #         if self.tokenizer.eos_id() in current_prompt_tokens:
    #             eos_idx=current_prompt_tokens.index(self.tokenizer.eos_id())
    #             current_prompt_tokens=current_prompt_tokens[:eos_idx]
    #         out_tokens.append(current_prompt_tokens)
    #         out_text.append(self.tokenizer.decode(current_prompt_tokens))
    #     return out_tokens,out_text
                
                 
    # def _sample_top_p(self, probs, p):
    #     # (B, vocab_size)
    #     probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    #     # (B, vocab_size)
    #     probs_sum = torch.cumsum(probs_sort, dim=-1)
    #     # (B, vocab_size)
    #     # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    #     mask = probs_sum - probs_sort > p 
    #     # Zero out all the probabilities of tokens that are not selected by the Top P
    #     probs_sort[mask] = 0.0 
    #     # Redistribute the probabilities so that they sum up to 1.
    #     probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    #     # Sample a token (its index) from the top p distribution
    #     next_token = torch.multinomial(probs_sort, num_samples=1)
    #     # Get the token position in the vocabulary corresponding to the sampled index
    #     next_token = torch.gather(probs_idx, -1, next_token) 
    #     return next_token

if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False  # Set this to True if you want to allow GPU usage
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"
 
    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision: 
        """
    ]
    try:
        model = LLaMa.build(
            checkpoints_dir='llama-2-7b',  # Ensure this directory exists
            tokenizer_path="tokenizer.model",  # Ensure this file exists
            load_model=True,  # Should be boolean, not a string
            max_seq_len=1024,
            max_batch_size=3,
            # max_batch_size=len(prompts),
            device=device
        )
        print("All Ok")
    except Exception as e:
        print(f"Error occurred: {e}")


    # out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    # assert len(out_texts) == len(prompts)
    # for i in range(len(out_texts)):
    #     print(f'{out_texts[i]}')
    #     print('-' * 50)
