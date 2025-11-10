
import os
import json
from datasets import Dataset, Image, DatasetDict
from huggingface_hub import create_repo



if __name__ == '__main__':

    hf_dataset_location = "harsha-desaraju/Telugu-text-image"

    fol_path = "correction_model/data"

    image_path = f"{fol_path}/generated/images"
    text_path = f"{fol_path}/generated/source_texts"
    files = [file for file in os.listdir(text_path) if file != '.DS_Store']
    files = sorted(files, key=lambda x: (int(x.split('.')[0].split('_')[1]), int(x.split('.')[0].split('_')[2])))


    split_len = 25000
    split_files = []
    i = 0
    while i*split_len<len(files):
        split_files.append(files[i*split_len:(i+1)*split_len])
        i += 1

        

    datasets = []
    
    for split_file in split_files:
        data_dct = {
            "image": [],
            "doc_id": [],
            "chunk_id": [],
            "src_text": [],
            "font": [],
            "ocr_text": []
        }
        for file in split_file:
            with open(f"{text_path}/{file}", 'r') as f:
                dct = json.load(f)
                img_name = dct['image_id']
                img_pth = f"{image_path}/{img_name}"
                _, doc_id, chunk_id = img_name.split('.')[0].split('_')
                data_dct['doc_id'].append(int(doc_id))
                data_dct['chunk_id'].append(int(chunk_id))
                data_dct['src_text'].append(dct['src_text'])
                data_dct['ocr_text'].append(dct['ocr_text'])
                font_name = ''.join(dct['font'].split('.')[:-1])
                data_dct['font'].append(font_name)
                data_dct['image'].append(img_pth)


        dataset = Dataset.from_dict(data_dct).cast_column("image", Image())
        datasets.append(dataset)

    dataset_dct = {}
    for i, dataset in enumerate(datasets):
        dataset_dct[str(i)] = dataset

    split_dataset = DatasetDict(dataset_dct)
    print(split_dataset)

    
    for key in split_dataset:
        dataset = split_dataset[key]
        dataset.push_to_hub(hf_dataset_location, split=key)






