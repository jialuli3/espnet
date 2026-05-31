# python hungarian_speaker_stitch.py \
#     /work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/genai_test_output/EN2002a.Mix-Headset_0.0s_300.0s.global_time.txt \
#     /work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/genai_test_output/EN2002a.Mix-Headset_240.0s_540.0s.global_time.txt \
#     --out_dir /work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/genai_test_output/stitch_out_ami

# python auto_detect_and_convert_timestamps.py /work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/genai_test_output/EN2002b*.txt --out_dir /work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/genai_test_output/global_time_stamps --drop_after_end

# python hungarian_multi_spk_stitch.py \
#    /work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/genai_test_output/global_time_stamps/EN2002b*.txt \
#    --out_dir /work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/genai_test_output/stitch_out_ami

# python transcript_to_rttm_drop_overlap.py /work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/genai_test_output/stitch_out_ami/EN2002b*global.txt \
#   --out_rttm /work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/genai_test_output/EN2002b.rttm \
#   --file_id EN2002b \
#   --overlap_sec 60 \
#   --merge_gap 0.2

spyder /work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/data/test/rttm /work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/genai_test_output/test.rttm -p -c 0.0 -u /work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/genai_test_output/test.uem > results.txt
