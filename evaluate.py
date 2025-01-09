from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
import torch
from accelerate import Accelerator
import pandas as pd

def evaluation(model,valid_dl, accelerator:Accelerator, valid_ids):

    progress_bar = tqdm(range(len(valid_dl)), disable=not accelerator.is_local_main_process)
    
    all_predictions = []
    all_references = []
    for batch in valid_dl:
        with torch.no_grad():
            logits, _ = model(**batch)
            logits = logits.reshape(-1)
            predictions = torch.sigmoid(logits)
            progress_bar.update(1)

        progress_bar.close()
        predictions, references = accelerator.gather_for_metrics((predictions,batch['labels'].to(torch.long).reshape(-1)))
        
        predictions = predictions.cpu().numpy().tolist()
        references = references.cpu().numpy().tolist()

        all_predictions.extend(predictions)
        all_references.extend(references)

    scores = compute_metric(predictions=predictions, references=references)

    results_df = pd.DataFrame()

    results_df['id'] = valid_ids
    results_df['predictions'] = all_predictions
    results_df['truths'] = all_references
    
    submit_df = results_df.copy()
    submit_df = submit_df.rename({'predicted':'generated'})
    submit_df = submit_df[['id','predictions']].copy()

    return {
        'submit_df':submit_df,
        'results_df':results_df,
        'scores_df':scores
    }
    

def compute_metric(predictions, references):

    assert len(predictions) == len(references)
    scores = roc_auc_score(references, predictions)
    return {'lb' : round(scores, 4)}
