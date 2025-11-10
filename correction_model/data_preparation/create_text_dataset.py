
import os
import json
from datasets import Dataset, DatasetDict
import pandas as pd




if __name__ == '__main__':

    hf_dataset_location = "harsha-desaraju/telugu-ocr-text"


    data_dct = {
        "doc_id": [],
        "chunk_id": [],
        "src_text": [],
        "ocr_text": []
    }

    fol_path = "correction_model/data"

    text_path = f"{fol_path}/generated/source_texts"
    files = [file for file in os.listdir(text_path) if file != '.DS_Store']
    files = sorted(files, key=lambda x: (int(x.split('.')[0].split('_')[1]), int(x.split('.')[0].split('_')[2])))

    
    for file in files:
        with open(f"{text_path}/{file}", 'r') as f:
            dct = json.load(f)
            img_name = dct['image_id']
            _, doc_id, chunk_id = img_name.split('.')[0].split('_')
            data_dct['doc_id'].append(int(doc_id))
            data_dct['chunk_id'].append(int(chunk_id))
            data_dct['src_text'].append(dct['src_text'])
            data_dct['ocr_text'].append(dct['ocr_text'])

    df = pd.DataFrame(data_dct)

    df = df.groupby(by='doc_id').agg({
        'src_text': ' '.join,
        'ocr_text': lambda x: " ".join(list(map(str.strip, x.tolist())))
    }).reset_index()


    data_dct = {
        'doc_id': df['doc_id'].tolist(), 
        'src_text': df['src_text'].tolist(), 
        'ocr_text': df['ocr_text'].tolist()
    }

    dataset = Dataset.from_dict(data_dct)
    print(dataset)

    dataset = DatasetDict({
        'train': dataset
    })

    dataset.push_to_hub(hf_dataset_location)

