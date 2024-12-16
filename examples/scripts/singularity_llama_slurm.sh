#!/bin/bash

#SBATCH -p spgpu              
#SBATCH -A chaijy2
#SBATCH -J openrlhf_tmp
#SBATCH -N 2                      # 64x8x4
#SBATCH -t 0-01:30:00             # wall time
#SBATCH --ntasks-per-node=1       # tasks per node
#SBATCH --exclusive                # exclusive node access
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --mem-per-gpu=20G
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --overcommit               # needed for pytorch
#SBATCH --output=out.log

# should be modified to train_sft_llama.sh, train_rm_llama.sh, train_dpo_llama, etc.
readonly training_script="train_ppo_llama_tmp.sh" 
readonly GPUS_PER_NODE=8

readonly PROJECT_PATH=$(cd ../../; pwd)
readonly IMAGE_NAME="pytorch_24.07-py3.sif"
readonly JOBLOG="$(pwd)/logs/$training_script-$SLURM_JOB_ID.log"

module load singularity
mkdir logs

# Job start
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# load training commands
source ./${training_script} slurm
echo training_commands &>> ${JOBLOG}
echo $training_commands &>> ${JOBLOG}

# master addr and port
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9904

singularity exec --nv --cleanenv --bind $PWD:/openrlhf --writable-tmpfs ${PROJECT_PATH}/${IMAGE_NAME} bash -c "
 export TMPDIR=/openrlhf/tmp
 mkdir -p \$TMPDIR
 pip install --user uninstall xgboost transformer_engine flash_attn 2>&1 | tee uninstall.log;
 pip install --user openrlhf 2>&1 | tee install_openrlhf.log;
 pip install --user git+https://github.com/OpenRLHF/OpenRLHF.git 2>&1 | tee git_install.log;
 ls
 pwd
 cd ../..
 HF_HOME=/scratch/chaijy_root/chaijy2/roihn/.cache/hugginface TORCH_EXTENSIONS_DIR=/scratch/chaijy_root/chaijy2/roihn/.cache/torch_extension \
 TRITON_CACHE_DIR=/scratch/chaijy_root/chaijy2/roihn/.cache/triton torchrun --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT -m ${training_commands} &>> ${JOBLOG}
"

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}