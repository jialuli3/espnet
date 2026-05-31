def read_file(file_path):
    with open(file_path, "r") as f:
        return f.readlines()

wav_lines=read_file("/work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/data_ihm_new/test/wav.scp")
print(wav_lines)
for line in wav_lines:
    print(line.strip().split()[0], line.strip().split()[1])
wav_dict = {line.strip().split()[0]: line.strip().split()[1] for line in wav_lines}
