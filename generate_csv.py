import numpy as np
import pandas as pd
import os

def get_time(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
        if 'seconds' in content:
            running_time = float(content.split('seconds')[0].strip())

            return running_time
        
def get_acc(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
        if 'percent' in content:
            accuracy = float(content.split('percent')[0].strip())
            return accuracy

txt_files=[fil for fil in os.listdir('.') if fil.endswith('.txt')]
sample_time_dict={}
acc_dict={}
for fil in txt_files:
    if 'sample' in fil:
        sample_time=get_time(fil)
        ipc_val=int(fil.split('_')[-1].split('_')[-1].split('_')[-1].split('_')[-1].split('.')[0])
        sample_time_dict[ipc_val]=sample_time

    elif 'testing' in fil:
        ipc_val=int(fil.split('_')[-1].split('_')[-1].split('.')[0])
        test_acc=get_acc(fil)
        acc_dict[ipc_val]=test_acc
    else:
        diff_time=get_time(fil)

sample_time_dict={key:val+diff_time for key,val in sample_time_dict.items()}

IPC=list(sample_time_dict.keys())
sample_time=list(sample_time_dict.values())
test_acc=[acc_dict[key] for key in IPC]


combined = list(zip(IPC, sample_time, test_acc))
sorted_combined = sorted(combined, key=lambda x: x[0])
IPC_sorted, sample_time_sorted, test_acc_sorted = zip(*sorted_combined)
IPC_sorted = list(IPC_sorted)
sample_time_sorted = list(sample_time_sorted)
test_acc_sorted = list(test_acc_sorted)

stat_data = {
    'Running Time': sample_time_sorted,
    'Test Accuracy': test_acc_sorted,
    'IPC': IPC_sorted
}

df = pd.DataFrame(stat_data)


df.to_csv('benchmark_minimax_diffusion.csv', index=False)
