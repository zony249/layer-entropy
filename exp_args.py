# Experiment Arguments; i.e., arguments pertaining to various experimental setups, rather than training or eval hyperparameters.
import os 
import sys 
from argparse import Namespace, ArgumentParser 
from typing import Dict, List, Tuple



def parse_exp_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--add_gist", action="store_true", help="Whether or not to inject gist tokens")
    parser.add_argument("--compression_rate", type=int, default=5, help="context:gist ratio")
    parser.add_argument("--attention_mask_mode", type=str, default="compression", choices=["compression", "full", "contextless"], help="compression mask is used for all compression scenarios")
    parser.add_argument("--compression_mode", default="none", choices=["none", "fourier", "average"], help="deterministic function used for compression. For learned compression, select 'none'")
    parser.add_argument("--gist_scheme", type=str, default="end", choices=["end", "dispersed"])
    parser.add_argument("--gist_granularity", type=int, default=1, help="Granularity of gist tokens under the dispersed scheme")

    ### DEPRECATED ###
    parser.add_argument("--entropy_model", type=str, default=None, help="Entropy model to help guide gist dispersion")
    parser.add_argument("--surprise_mode", type=str, default="entropy", choices=["entropy", "ce"])
    parser.add_argument("--entropy_model_temp", type=float, default=1)
    ### END DEPRECATED ###

    parser.add_argument("--act_guided_chunking", type=str, default="none", choices=["none", "normdiff"], 
                        help="""activation-guided chunking.
                            normdiff: chunking based on norm of difference vectors between tokens
                        """)
    parser.add_argument("--chunking_model", type=str, default=None, help="Chunking model used to help guide gist dispersion")


    args, unknown = parser.parse_known_args()
    return args

def join_args(*list_args: List[Namespace]) -> Namespace: 
    joint_dict = {k: v for a in list_args for k, v in vars(a).items()}
    return Namespace(**joint_dict)


if __name__ == "__main__":
    parse_exp_args()