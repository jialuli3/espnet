def read_file(file_name):
    with open(file_name, "r") as f:
        return f.readlines()

def write_file(content, file_name):
    with open(file_name, "w") as f:
        f.writelines(content)

def add_na(input_file,out_file):
    content=read_file(input_file)
    for i in range(len(content)):
        content[i]=content[i].strip().split("\t")
        content[i].append("<NA>\n")
        content[i]=" ".join(content[i])
    write_file(content,out_file)

for folder in ["test","train"]:
    add_na(f"data_librimix/{folder}/rttm", f"data_librimix/{folder}/rttm_")
