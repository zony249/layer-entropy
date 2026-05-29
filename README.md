# Layer Entropy
Note: Instructions may change, as this codebase is constantly changing.
## Setup
Software Requirements: 
- Python 3.11
- accelerate 1.12 
- transformers 4.57 
- datasets 4.5.0 
- evaluate 0.4.6
- pyarrow 21.0.0

## Steps to run
1. Download model, if needed: 
```bash
$ hf download Qwen/Qwen3-0.6B --local_dir Qwen/Qwen3-0.6B 
```

2. Train compression model:
    - Scripts are located in `scripts/dynamic_gist_compress/`. 
    - You may want to change certain paths, such as model path, where to dump the outputs, etc. 
    - Experiment-specific arguments are found in `exp_args.py`. 
    - run `scripts/dynamic_gist_compress/<any script in any subfolder>`. The program will ask you to login to wandb for logging. Right now it's mandatory, but in the future I may make it opt-in. 

3. Evaluate compression model: 
    - Set the path for trained compression model in `scripts/eval_squad.sh` 
    - Run `scripts/eval_squad.sh`