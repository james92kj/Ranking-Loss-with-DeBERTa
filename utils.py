from accelerate import Accelerator
import torch
import shutil

# picking the head learning rate 
def get_lr(opt): 
    return opt.param_groups[0]['lr'] * 1e6



class LossMeter:

    def __init__(self):
        self.reset() 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n 
        self.count += n 
        self.avg = self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0
        self.val = 0


def save_checkpoint(ckpt_info, cfg, accelerator:Accelerator):
    name = cfg.output.name
    filename = f"{cfg.output.dir}/{name}_last_pt.rar"
    
    if accelerator.is_local_main_process:
        torch.save(ckpt_info,filename,_use_new_zipfile_serialization=False)

        if ckpt_info['is_best']:
            shutil.copyfile(filename,f'{cfg.output.dir}/{name}_best.rar')
            accelerator.print('##### File Copied ######')
    