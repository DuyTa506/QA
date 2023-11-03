import torch
from transformers import AutoModel, AutoTokenizer,AutoModelForSeq2SeqLM
from pyvi.ViTokenizer import tokenize
from datasets import load_dataset, DatasetDict, load_from_disk
token  = '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'

PhobertTokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder', token = token)
model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder', token = token)
corrector = AutoModelForSeq2SeqLM.from_pretrained("bmd1905/vietnamese-correction")
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def mean_pooling_with_aligned(model_output, use_all_ones_mask=True):
    token_embeddings = model_output  
    input_mask = torch.ones_like(token_embeddings)
    input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()


    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embeddingvector(text):
    inputs = PhobertTokenizer(
        text, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        model_output = model(**inputs)
        embeddings = mean_pooling(model_output, inputs['attention_mask'])
    return embeddings

def embeddingvector_with_alignment(text):
    with torch.no_grad():
        inputs = PhobertTokenizer(
            text, padding=True, truncation=True, return_tensors="pt")

        correct_out = corrector.generate(inputs["input_ids"])
        end_correct= PhobertTokenizer.batch_decode(correct_out, skip_special_tokens=True)
        aligned = ' '.join(end_correct)
        inputs = PhobertTokenizer(
            text, padding=True, truncation=True, return_tensors="pt")
        model_output = model(**inputs)
        embeddings = mean_pooling(model_output, inputs['attention_mask'])
    return embeddings

def load_data(path):
    from datasets import load_dataset, DatasetDict, load_from_disk
    processed_dataset = load_from_disk(path)
    print("Đang khởi động ứng dụng, vui lòng đợi trong giây lát !!")
    embeddings_dataset = processed_dataset.map(
        lambda x: {"embeddings": embeddingvector(x["Câu hỏi"]).numpy()[0]}, remove_columns = ['__index_level_0__'], keep_in_memory=True
    )
    embeddings_dataset.add_faiss_index(column="embeddings")

    return embeddings_dataset



def return_answer(question,threshold1,threshold2):
  """
  Type all function here bro, end to end from me !
  """
  from sentence_transformers import util
  embed_question = embeddingvector_with_alignment(question).numpy()
  scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", embed_question, k=3
  )
  cosine = [util.cos_sim(embed_question,i) for i in samples["embeddings"]]
  cosine = [float(tensor.item()) for tensor in cosine]
  out(question,samples,threshold1,threshold2,cosine)
  return


def out(sentence, samples, t1, t2, scores):

    result = [answer for answer, score in zip(samples['Trả lời'], scores) if score > t1]
    best_answer = result[0] if len(result) >= 1 else None
    if len(result) == 0:
        result = [answer for answer, score in zip(samples['Trả lời'], scores) if score >= t2]
    
    print(f"Câu hỏi của bạn: \n{sentence}")
    if best_answer is not None :
        print(f"Câu trả lời tốt nhất: \n{best_answer}")
    if len(result) >0 :
      print("Các câu trả lời khả năng:\n")
      for index, answer in enumerate(result):
          if index > 0:
              print(answer, "\n")
    else :
      print("Không tìm thấy câu trả lời, đang chuyển sang bộ phận CSKH")

embeddings_dataset = load_data('./qa')
return_answer("Anh muốn rút tiền", 0.5,0.1)