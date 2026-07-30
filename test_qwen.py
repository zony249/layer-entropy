import os 
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
import torch
from torch.nn import functional as F

from data_utils.squad import Squad
from utils import GistDataCollator

from models.modeling_qwen3 import (CompQwen3ForCausalLM, ZipQwen3ForCausalLM)
from utils import ChunkCollator

if __name__ == "__main__": 

    print("starting debugger")
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()

    squad = Squad(["train", "validation"], batch_size=1, preprocess_validation=False)

    trainset = squad.get_dataset("train")
    valset = squad.get_dataset("validation").select(range(100))


    model = ZipQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", attn_implementation="eager").cuda()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # tok = model.enable_compression(tok)
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
                return_tensors="pt").to(model.device)




    token_type_ids = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])
    probabilities = F.one_hot(token_type_ids, num_classes=3)
    output = model.generate(**sents, chunk_signal=probabilities, max_new_tokens=10)
    print(tok.batch_decode(output))
    pass