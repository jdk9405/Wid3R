import sys
import os

# # --- Start per-rank log redirection ---
# def _redirect_logs():
#     # Environment variable set for each process by Accelerate or torchrun
#     local_rank = os.environ.get("LOCAL_RANK", "unknown")

#     # Use "w" to overwrite; use "a" to append logs across multiple retries
#     stdout_file = open(f"rank_{local_rank}.out", "w")
#     stderr_file = open(f"rank_{local_rank}.err", "w")

#     sys.stdout = stdout_file
#     sys.stderr = stderr_file

# _redirect_logs()
# --- End per-rank log redirection ---

sys.path.append('.')

import hydra
import trainers
import ipdb

@hydra.main(version_base="1.2", config_path="../configs", config_name="default")
def main(hydra_cfg):
    trainer = eval(hydra_cfg.trainer)(hydra_cfg)
    trainer.train()

if __name__ == '__main__':
    main()
