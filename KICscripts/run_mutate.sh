#!/bin/bash
#SBATCH --job-name train_task                  # 任务在 squeue 中显示任务名为 example
#SBATCH --output %A_%a.out                  # 任务输出重定向至 [任务组id]_[组内序号].out
#SBATCH --time 1-1:00:00                    # 任务最长运行 1 天 1 小时，超时任务将被杀死
#SBATCH --array 0                        # 提交 16 个任务，组内序号分别为 0,1,2,...15
#SBATCH --mail-user nohandsomewujun@gmail.com       # 将任务状态更新以邮件形式发送至 example@gmail.com
#SBATCH --mail-type ALL                     # 任务开始运行、正常结束、异常退出时均发送邮件通知
#SBATCH --cpus-per-task 50

# 任务 ID 通过 SLURM_ARRAY_TASK_ID 环境变量访问
# 上述行指定参数将传递给 sbatch 作为命令行参数
# 中间不可以有非 #SBATCH 开头的行

# 执行 sbatch 命令前先通过 conda activate [env_name] 进入环境
bash mutate.sh
