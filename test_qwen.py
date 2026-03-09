import os 
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
import torch

from data_utils.squad import Squad
from utils import GistDataCollator

from models.modeling_qwen3 import CompQwen3ForCausalLM

if __name__ == "__main__": 

    try: 
        debug_mode = int(os.environ["DEBUG_MODE"]) 
        if debug_mode > 0: 
            print("starting debugger")
            import debugpy
            debugpy.listen(("172.26.93.143", 5679))
            print("Waiting for debugger attach...")
            debugpy.wait_for_client()
    except: 
        pass

    squad = Squad(["train", "validation"])

    trainset = squad.get_dataset("train")
    valset = squad.get_dataset("validation").select(range(100))


    model = CompQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to("cuda")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tok = model.enable_compression(tok)
    # collator = GistDataCollator(tok, -1, add_gist=False)
    
    # batch = [valset[i] for i in range(4)]
    # batch = collator(batch).to("cuda")
    

    # shift left: 
    # batch["labels"] = torch.cat((batch["labels"][:, 1:], (torch.ones((4, 1)).cuda() * tok.eos_token_id).long()), dim=1)
    # model(**batch)

    sents = tok(["what is the largest planet in the solar system?", 
                 "lmao what the heck is this..."], 
                padding=True, 
                padding_side="left", 
                return_tensors="pt").to("cuda")
    output = model.generate(**sents, use_cache=True)
    pass