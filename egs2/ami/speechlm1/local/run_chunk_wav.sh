#!/bin/bash

# sbatch --time=4:00:00 -p RM-shared --ntasks=1 --cpus-per-task=16 --mem=60000M --account=bbjs-delta-cpu run_chunk_wav.sh

python ./chunk_wav.py /work/nvme/bbjs/jialuli3/test_wav/ami_wav/EN2002b.Mix-Headset.wav /work/nvme/bbjs/jialuli3/test_wav/ami_wav_segments --overlap_sec 60