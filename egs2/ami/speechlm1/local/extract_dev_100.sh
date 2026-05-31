#sort wav.scp
raw_dir=/work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/dump_synthetic/raw_codec_ssl_sd_event_sad_od_dur30_skip10_alimeeting/dev_100
dev_dir=/work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/data_synthetic/dev
dev_100_dir=/work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/data_synthetic/dev_100

# sort -t '_' -k7,7 -n $input_dir/wav.scp > $output_dir/wav.scp

#make json

awk '
NR==FNR {
    id=$1
    sub(/-[^-]+-[^-]+$/, "", id)
    keep[id]=1
    next
}
{
    id=$2
    sub(/-[^-]+-[^-]+$/, "", id)
}
(id in keep)
' $raw_dir/wav.scp $dev_dir/rttm > $dev_100_dir/rttm
