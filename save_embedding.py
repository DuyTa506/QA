import json
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoModel, AutoTokenizer,AutoModelForSeq2SeqLM
from pyvi.ViTokenizer import tokenize
from pyvi import ViUtils
from corrector import check_spel
from datasets import load_dataset, DatasetDict, load_from_disk
token  = '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'

PhobertTokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder', token = token)
model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder', token = token)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embeddingvector(text):
    inputs = PhobertTokenizer(
        text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
        embeddings = mean_pooling(model_output, inputs['attention_mask'])
    return embeddings

with open("all_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)
qa_pairs = []
for q, a in data.items():
    qa_pairs.append({"Câu hỏi": q.lower(), "Trả lời": a})
df = pd.DataFrame(qa_pairs)
dataset = Dataset.from_pandas(df)

# def embeddingvector(text):
#     inputs = PhobertTokenizer(
#         text, padding=True, truncation=True, return_tensors="pt")
#     with torch.no_grad():
#         embeddings = model(**inputs, output_hidden_states=True,
#                            return_dict=True).pooler_output
#     return embeddings


def save_data(dataset):
    from datasets import load_dataset, DatasetDict, load_from_disk
    print("Embedding vector data !!")
    embeddings_dataset = dataset.map(
        lambda x: {"embeddings": embeddingvector(x["Câu hỏi"]).numpy()[0]}, keep_in_memory=True
    )
    embeddings_dataset.save_to_disk(
        "./embeddingbkai", num_proc=1
    )
    return True

save_data(dataset)