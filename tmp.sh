
#Tensorboard
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork
tensorboard --logdir "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/output/sub10/sub10-gravity-9-keypoint-7-balance/summaries" --port 6006 --bind_all
tensorboard --logdir "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/Human2Robot/intermimic-fork/output/sub10/sub10-gravity-9-keypoint-7-no-balance/summaries" --port 6007 --bind_all

#tmux setup
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA_Quantization/chi/miniforge3/etc/profile.d/conda.sh
conda activate intermimic-gym
sh isaacgym/scripts/train_g1_native_sub10.sh sub10-gravity-7-keypoint-7 --headless