
#!/bin/bash

# EDITABLE PARAMS
   logdir="./"
   name="Training-ECAPA-VoxCeleb-256"

   # CONFIG FILE
   YAML="/fs/scratch/users/francesco_nespoli/SpeechBrain/train_on_voxceleb/training/hparams_256.yaml"

# FIXED
   container="/fs/scratch/users/francesco_nespoli/SpeechBrain/speechbrain.sif"

# HARDWARE
   gpus=4
   gputype=v100

# PREPARE OUTPUT FILES/DIRS
   mkdir -p $logdir/stderr_${name}
   mkdir -p $logdir/stdout_${name}

# TRAINING COMMAND

   sbatch -p gpu --array=1-1 --job-name=$name --gres=gpu:${gputype}:${gpus} --mem=32g --cpus-per-task=8 --time=15-15:00:00  -o $logdir/stdout_${name}/${name}.out -e $logdir/stderr_${name}/${name}.err ./run_fine_tuning.sh $YAML ${container}

### --nodelist=euent-gpu2-pg0-3



































#   sbatch --error=$logdir/log/job%j.out --output=$logdir/log/job%j.err --array=1-${num} extract_embedding.sh ${audio_list} $outdir $source $savedir --cpus_per_task=2
