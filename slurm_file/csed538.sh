#!/bin/bash
#SBATCH --job-name=csed538_job      # 작업 이름 설정

##SBATCH --time=00:30:00             # 작업 실행 시간 설정 (시:분:초)
#SBATCH --output=csed538.%j.out     # 작업 실행 결과 출력 파일명 설정
#SBATCH --error=csed538.%j.err      # 작업 실행 에러 출력 파일명 설정

#SBATCH -p titanxp                  # queue name or partition
#SBATCH --gres=gpu:2                # gpus per node

#SBATCH --nodes=1                   # 노드 수 설정
#SBATCH --tasks-per-node=1          # 각 노드 당 작업 수 설정
#SBATCH --cpus-per-task=4           # 각 작업에 할당된 CPU 코어 수 설정

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "HOME=$HOME"
PROJECT_DIR=$HOME/POSTECH-CSED538
echo "PROJECT_DIR=$PROJECT_DIR"

srun -I /bin/hostname
srun -I /bin/pwd
srun -I /bin/date

module purge
module load cuda/11.3
module load slurm/21.08.4

echo "source $HOME/anaconda3/etc/profile.d/conda.sh"
source $HOME/anaconda3/etc/profile.d/conda.sh   # add conda to PATH

echo "conda activate ddpm-cd"
conda activate ddpm-cd

echo "start training"
# python ddpm_train.py --config config/levir.json -enable_wandb -log_eval
python $PROJECT_DIR/ddpm_train.py --config $PROJECT_DIR/config/levir.json -enable_wandb -log_eval

echo "conda deactivate"
conda deactivate

date
squeue --job $SLURM_JOBID

echo "### END ###"