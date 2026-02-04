import os
import time
import sys
import subprocess

class GPUGet:
    def __init__(self,
                 min_gpu_number,
                 time_interval):
        self.min_gpu_number = min_gpu_number
        self.time_interval = time_interval

    def get_gpu_info(self):
      
        gpu_status = os.popen('nvidia-smi | grep %').read().split('|')[1:]
        gpu_dict = dict()
        for i in range(len(gpu_status) // 4):
            index = i * 4
            gpu_state = str(gpu_status[index].split('   ')[2].strip())
            gpu_power = int(gpu_status[index].split('   ')[-1].split('/')[0].split('W')[0].strip())
            gpu_memory = int(gpu_status[index + 1].split('/')[0].split('M')[0].strip())
            gpu_dict[i] = (gpu_state, gpu_power, gpu_memory)
        return gpu_dict

    def loop_monitor(self):
        available_gpus = []
        while True:
            gpu_dict = self.get_gpu_info()
            print(f"\nChecking GPU states...") 
            for i, (gpu_state, gpu_power, gpu_memory) in gpu_dict.items():
            
                gpu_str = f"GPU{i}: State: {gpu_state}, Power: {gpu_power}W, Memory: {gpu_memory}MiB"
                print(gpu_str)

            
                if gpu_state == "P8" and gpu_power <= 40 and gpu_memory <= 1000:
          
                    print(f"GPU{i} is available (state: {gpu_state}, power: {gpu_power}W, memory: {gpu_memory}MiB)")
                    available_gpus.append(i)
            
            if len(available_gpus) >= self.min_gpu_number:
                print(f"Found {len(available_gpus)} available GPUs, ready to start training.")
                return available_gpus
            else:
                print(f"Not enough GPUs found. Current available: {len(available_gpus)} / Required: {self.min_gpu_number}.")
                available_gpus = []
                time.sleep(self.time_interval)

    def run(self, gpu_list_str, cmd_parameter, cmd_command):

        full_command = f"""bash -i -c 'cd /home/luzhenwei/chl/0pix2pix/pix2pix_qrcode_modify/ && conda activate pytorch113 && {cmd_parameter} {cmd_command} --gpu_ids {gpu_list_str} &'"""
        

        print(f"\nExecuting command: {full_command}")
        

        subprocess.run(full_command, shell=True, executable="/bin/bash")


if __name__ == '__main__':
    min_gpu_number = 3 
    time_interval = 5
    gpu_get = GPUGet(min_gpu_number, time_interval)


    cmd_parameter = "" 
    cmd_command = "nohup python /home/luzhenwei/chl/0pix2pix/pix2pix_qrcode_modify/train.py --dataroot /home/luzhenwei/chl/0pix2pix/pix2pix_qrcode/datasets/qrcodes/AB --name qr1 --niter 300 --niter_decay 100 --batch_size 256 --continue_train --epoch_count 301"


    available_gpus = gpu_get.loop_monitor()
    gpu_list_str = ",".join(map(str, available_gpus))


    gpu_get.run(gpu_list_str, cmd_parameter, cmd_command)
