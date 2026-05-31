mkdir -p data_librimix/dev
mkdir -p data_librimix/test
mkdir -p data_librimix/train

for file_name in reco2dur rttm segments spk2utt utt2spk wav.scp; do
    cat data_librimix/dev2/${file_name} data_librimix/dev3/${file_name} > data_librimix/dev/${file_name} 
done
