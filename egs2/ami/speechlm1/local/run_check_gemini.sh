#!/bin/bash

# sbatch --time=2-0:00:00 -p cpu --ntasks=1 --cpus-per-task=16 --mem=60000M --account=bbjs-delta-cpu run_check_gemini.sh

python ./check_gemini_output.py