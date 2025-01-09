from datasets import Dataset
from transformers import AutoTokenizer

class AiDataset:

    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer =  AutoTokenizer.from_pretrained(
            cfg.model.backbone_path
        )
    
    '''
        tokenize_dataset takes 
        examples in batched form 
        and the input_ids and 
        attention_mask 
    '''
    def tokenize_dataset(self, examples):

        tzd =  self.tokenizer(
            examples['text'],
            max_length=self.cfg.model.max_length,
            padding=False,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=True
        )

        return tzd

    def get_dataset(self, df):
        '''
            The function takes the dataframe 
            and return a Dataset instance of
            Pytorch
        '''
        copied_df = df.copy()
        data = Dataset.from_pandas(df=copied_df)
        td = data.map(self.tokenize_dataset, batched=True)
        return td


