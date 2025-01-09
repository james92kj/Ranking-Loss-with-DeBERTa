from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from evaluate import evaluation as run_evaluation
from utils import get_lr, LossMeter, save_checkpoint
from tqdm.auto import tqdm
import pandas as pd
import datasets
import transformers
import random
import hydra
import os
import torch

try:
    from ai_dataset import AiDataset
    from ai_collator import TrainCollator, ValidCollator
    from ai_optimizer import get_optimizer
    from ai_model import AiModel
except:
    raise ImportError



@hydra.main(config_path='./conf', config_name='cfg')
def main(cfg):

    if cfg.use_wandb:
        accelerator = Accelerator(
            log_with='wandb',
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps
        )

        accelerator.init_trackers(
            project_name='ranking_loss',
            config=OmegaConf.to_container(cfg=cfg,resolve=True)
        )

    else:
        accelerator=Accelerator(
            gradient_accumulation_plugin=cfg.train.gradient_accumulation_steps
        )

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        accelerator.print(prefix + unit*50 + suffix)

    # --------------------------------------- Enable logging ------------------------------------------
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    print_line()

    # --------------------------------------- Output ------------------------------------------
    if accelerator.is_local_main_process:
        os.makedirs(cfg.output.dir,exist_ok=True)
    
    # --------------------------------------- set the seed ------------------------------------------
    accelerator.print(f'Setting the seed {cfg.seed}')
    set_seed(cfg.seed)
    print_line()
    # --------------------------------------- load the data -------------------------------------------
    # read the dataframe
    df = pd.read_csv(os.path.join(cfg.input_dir,'train_essays.csv'))
    # remove all records which are empty in text column 
    df = df[~df['text'].isna()].copy()
    # reset the index 
    df = df.reset_index(drop=True)

    accelerator.print(f'Loaded {len(df)} records. The head portion is {df.head}')
    print_line()
    # --------------------------------------- Data split -----------------------------------------------
    rng = random.Random(cfg.seed)
    df['fold'] = df['text'].apply(lambda x: 'train' if rng.random() < 0.99 else 'valid')
    train_df = df[df['fold'] == 'train'].copy()
    valid_df = df[df['fold'] == 'valid'].copy()

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)


    #extract the prompt ids 
    prompt_ids =  train_df['prompt_id'].unique().tolist()
    gdf = train_df.groupby('prompt_id')['id'].apply(list).reset_index()
    prompt2ids = dict(zip(gdf['prompt_id'], gdf['id']))


    # print the shape of data 
    accelerator.print(f'Shape of data train-{len(train_df)}, valid-{len(valid_df)} ')
    print_line()


    #  --------------------------------------- Dataset ----------------------------------------------
    with accelerator.main_process_first():
        ai_dataset = AiDataset(cfg=cfg)

        train_dataset = ai_dataset.get_dataset(df=train_df)
        valid_dataset = ai_dataset.get_dataset(df=valid_df)

    tokenizer = ai_dataset.tokenizer

    # --------------------------------------- Set columns ----------------------------------------------

    train_dataset.set_format(
        type=None,
        columns=[
            'id',
            'input_ids',
            'attention_mask',
            'generated'
        ]
    )

    valid_dataset.set_format(
        type=None,
        columns=[
            'id',
            'input_ids',
            'attention_mask',
            'generated'
        ]
    )

    valid_ids = valid_df['id']
    #  --------------------------------------- Collator ----------------------------------------------
    kwargs = dict(
        train_ds = train_dataset,
        prompt_ids = prompt_ids,
        prompt2ids = prompt2ids
    )
    
    # test the difference between padding and pad_to_multiple_of

    train_collator = TrainCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=64,
        kwargs=kwargs
    )

    valid_collator = ValidCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=64
    )

    # --------------------------------------- Data loader ----------------------------------------------
    train_dl = DataLoader(
        train_dataset,
        batch_size=cfg.train.per_device_train_batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_collator
    )

    valid_dl = DataLoader(
        valid_dataset,
        batch_size=cfg.train.per_device_eval_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=valid_collator
    )

    # --------------------------------------- Model ------------------------------------------------------
    model = AiModel(cfg)

    # --------------------------------------- Optimizer & Scheduler --------------------------------------------------
    optimizer = get_optimizer(cfg=cfg,model=model)


    gradient_accumulation_steps = cfg.train.gradient_accumulation_steps
    num_epochs = cfg.train.num_train_epochs
    num_updates_per_epoch = len(train_dl) // gradient_accumulation_steps
    num_training_steps = num_epochs * num_updates_per_epoch
    num_warmup_steps = int(cfg.train.warmup_pct * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )


    # prepare the model, dataloader, optimizer
    model, optimizer, train_dl, valid_dl, scheduler = accelerator.prepare(model,optimizer,train_dl,valid_dl,scheduler)

    # ------------------------------------ Training Setup ------------------------------------------------
    current_iteration = 0
    best_lb = 1e6
    patience_tracker = 0
    start_epoch= 0 
    progress_bar = None

    accelerator.wait_for_everyone()
    # ------------------------------------ Training loop ------------------------------------------------

    for epoch in range(start_epoch,num_epochs):
        
        if progress_bar is not None and epoch != 0:
            progress_bar.close()

        model.train()
        progress_bar = tqdm(range(num_updates_per_epoch), disable=not accelerator.is_local_main_process)
        epoch_loss = LossMeter()
        
        for step, batch in enumerate(train_dl):

            with accelerator.accumulate(model):
                _, loss = model(**batch)
                accelerator.backward(loss)
            
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=cfg.train.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()


                epoch_loss.update(loss.item())

            if accelerator.sync_gradients:
                step_info = f'STEP. {(current_iteration % num_updates_per_epoch)+1:5}/{num_updates_per_epoch:5}.'
                lr_info = f'LR. {get_lr(opt=optimizer):.4f}'
                loss_info = f'LOSS. {epoch_loss.avg:.4f}'

                progress_bar.set_description(step_info + lr_info + loss_info)
                progress_bar.update(1)

                current_iteration += 1
            
            if accelerator.sync_gradients and current_iteration % cfg.train.eval_frequency == 0:
                
                model.eval()
        
                eval_response = run_evaluation(model=model,valid_dl=valid_dl,accelerator=accelerator,valid_ids=valid_ids)

                lb_score = eval_response['scores_df']['lb']
                msg = f'Epoch- {epoch} | Step - {step} | Current Iteration {current_iteration} | lb_score {lb_score}'
                accelerator.print(msg)
            
                is_best=False 
                if lb_score <= best_lb:
                    best_lb = lb_score
                    is_best = True
                    patience_tracker = 0
                else:
                    patience_tracker += 1


                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    'epoch':epoch,
                    'is_best':is_best,
                    'optim_state_dict': optimizer.state_dict(),
                    'state_dict': unwrapped_model.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'current_iteration': current_iteration+1,
                    'lb_score':lb_score
                }
                
                save_checkpoint(checkpoint,accelerator=accelerator,cfg=cfg)
                
                model.train()
                torch.cuda.empty_cache()

                if patience_tracker >= cfg.train.no_improvement_threshold:
                    model.eval()
                    accelerator.print(f'No improvement in evaluation score. quit training')
                    return

if __name__ == '__main__':
    main()
