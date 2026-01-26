
import os
import json
from pathlib import Path
import pandas as pd
import re


def clean_ocr_text(text):
        # Replace single newlines with spaces
        # Preserve double newlines (paragraphs)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        return text.strip()


if __name__ == '__main__':

    # fol_path = "correction_model/data/generated/source_texts"
    # paths = list(Path(fol_path).rglob('*.json'))

    # df = {
    #     "src_text": [],
    #     "doc_id": [],
    #     "chunk_id": [],
    #     "ocr_text": []
    # }

    # for path in paths:
    #     file_name = path.name
    #     _, doc_id, chunk_id = file_name.split('.')[0].split('_')
    #     with open(path, 'r') as f:
    #         dct = json.load(f)
    #         df['src_text'].append(dct['src_text'])
    #         df['ocr_text'].append(dct['ocr_text'])
    #         df['doc_id'].append(int(doc_id))
    #         df['chunk_id'].append(int(chunk_id))

    df = pd.read_parquet(f"correction_model/data/ocr_text.parquet")
    print(df)

    