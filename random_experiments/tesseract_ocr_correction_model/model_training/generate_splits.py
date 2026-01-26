
import os
import json
import numpy as np
from datasets import Dataset, Image, DatasetDict




if __name__ == '__main__':

    hf_dataset_location = "harsha-desaraju/Telugu-text-image"

    fol_path = "correction_model/data"

    image_path = f"{fol_path}/generated/images"
    text_path = f"{fol_path}/generated/source_texts"
    files = [file for file in os.listdir(text_path) if file != '.DS_Store']
    files = sorted(files, key=lambda x: (int(x.split('.')[0].split('_')[1]), int(x.split('.')[0].split('_')[2])))        

    dataset = []
    for file in files:
        data_dct = {}
        with open(f"{text_path}/{file}", 'r') as f:
            dct = json.load(f)
            img_name = dct['image_id']
            _, doc_id, chunk_id = img_name.split('.')[0].split('_')
            text = f"""OCR text:{dct['ocr_text']}\n\nCorrect text:{dct['src_text']}"""
            data_dct['text'] = text
        dataset.append(data_dct)

   
    np.random.seed(42)

    inds = np.random.permutation(len(dataset))
    val_split = 0.05
    test_split = 0.05
    
    val_data = []
    val_inds = inds[:int(val_split*len(dataset))]
    for ind in val_inds:
        val_data.append(dataset[ind])

    test_data = []
    test_inds = inds[int(val_split*len(dataset)): int((val_split+test_split)*len(dataset))]
    for ind in test_inds:
        test_data.append(dataset[ind])

    train_data = []
    train_inds = inds[int((val_split+test_split)*len(dataset)):]
    for ind in train_inds:
        train_data.append(dataset[ind])

    print(f"Num train samples: ", len(train_data))
    print(f"Num test samples: ", len(test_data))
    print(f"Num val samples: ", len(val_data))


    with open("correction_model/model_training/data/train.jsonl", 'w') as f:
        for dct in train_data:
            json.dump(dct, f, ensure_ascii=False)
            f.write('\n')

    with open("correction_model/model_training/data/valid.jsonl", 'w') as f:
        for dct in val_data:
            json.dump(dct, f, ensure_ascii=False)
            f.write('\n')

    with open("correction_model/model_training/data/test.jsonl", 'w') as f:
        for dct in test_data:
            json.dump(dct, f, ensure_ascii=False)
            f.write('\n')

    












