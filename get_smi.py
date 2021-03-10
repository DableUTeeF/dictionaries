import re
import subprocess


def get_smi():
    command = ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.total,memory.used,temperature.gpu', '--format=csv']
    p = subprocess.check_output(command)
    data = p.decode('utf-8').split('\n')[1].split(',')
    name = data[0]
    util = int(re.findall(r'\d+', data[1])[0])
    ram_total = int(re.findall(r'\d+', data[2])[0])
    ram_used = int(re.findall(r'\d+', data[3])[0])
    temp = int(re.findall(r'\d+', data[4])[0])
    ram_percent = int(ram_used) / int(ram_total)

    return temp, util, ram_used, ram_total


if __name__ == '__main__':
    print(get_smi())
