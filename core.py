import torch
from transformers import AutoModel, AutoTokenizer,AutoModelForSeq2SeqLM
from pyvi.ViTokenizer import tokenize
from datasets import load_dataset, DatasetDict, load_from_disk
import lite_corrector as corrector
# PhobertTokenizer = AutoTokenizer.from_pretrained(
#     "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
# model = AutoModel.from_pretrained(
#     "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")


# def embeddingvector(text):
#     inputs = PhobertTokenizer(
#         text, padding=True, truncation=True, return_tensors="pt")
#     with torch.no_grad():
#         embeddings = model(**inputs, output_hidden_states=True,
#                            return_dict=True).pooler_output
#     return embeddings


token  = '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'

PhobertTokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder', token = token)
model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder', token = token)

#corrector = AutoModelForSeq2SeqLM.from_pretrained("bmd1905/vietnamese-correction") Replace with lite corrector and trigram LM
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

def load_data(path):
    from datasets import load_dataset, DatasetDict, load_from_disk
    processed_dataset = load_from_disk(path)
    print("Đang khởi động ứng dụng, vui lòng đợi trong giây lát !!")
    embeddings_dataset = processed_dataset.map(
        lambda x: {"embeddings": embeddingvector(x["Câu hỏi"]).numpy()[0]}, remove_columns=['__index_level_0__'], keep_in_memory=True
    )
    embeddings_dataset.add_faiss_index(column="embeddings")
    return embeddings_dataset

def load_embedding(path):
    from datasets import load_dataset, DatasetDict, load_from_disk
    processed_dataset = load_from_disk(path)
    print("Đang khởi động ứng dụng, vui lòng đợi trong giây lát !!")
    print(processed_dataset)
    processed_dataset.add_faiss_index(column="embeddings")
    return processed_dataset

def embeddingvector_with_alignment(text):
    with torch.no_grad():
        inputs = PhobertTokenizer(
            text, padding=True, truncation=True, return_tensors="pt")

        correct_out = corrector.generate(inputs["input_ids"])
        end_correct= PhobertTokenizer.batch_decode(correct_out, skip_special_tokens=True)
        aligned = ' '.join(end_correct)
        print(aligned)
        input_aligned = PhobertTokenizer(
            aligned, padding=True, truncation=True, return_tensors="pt")
        model_output = model(**input_aligned)
        embeddings = mean_pooling(model_output, input_aligned['attention_mask'])
    return embeddings

def lite_align_embedding(text):
    with torch.no_grad():
        aligned = corrector.correct(text)
        input_aligned = PhobertTokenizer(
            aligned, padding=True, truncation=True, return_tensors="pt")
        model_output = model(**input_aligned)
        embeddings = mean_pooling(model_output, input_aligned['attention_mask'])
    return embeddings

def return_answer_aligned_lite(question, upper_threshold, lower_threshold, embeddings_dataset):
    ans = []
    scs = []
    from sentence_transformers import util
    try:
        embed_question = lite_align_embedding(question).numpy()
        l2_score, samples = embeddings_dataset.get_nearest_examples(
            "embeddings", embed_question, k=3
        )
        #l2_similarity = [1 / (1 + distance) for distance in l2_score]
        cosine = [util.cos_sim(embed_question, i) for i in samples["embeddings"]]
        #print(l2_score)
        scores = [float(tensor.item()) for tensor in cosine]
        #print(l2_similarity)
        #scores = [0.2 * l2 + 0.8 * cosine for l2, cosine in zip(l2_similarity, scores)]

        result = [answer for answer, score in zip(samples['Trả lời'], scores) if score > upper_threshold]

        if len(result) >= 1:
            print("Câu trả lời chuẩn xác là:")
            ans.append(result[0])
            scs.append(scores[0])
            return ans, scs
        elif len(result) == 0:
            result = [answer for answer, score in zip(samples['Trả lời'], scores) if score >= lower_threshold]
            print(len(result))
            if len(result) > 0:
                for answer, score in zip(result, scores):
                    #print("Câu trả lời khả năng là:")
                    ans.append(answer)
                    scs.append(score)
                return ans, scs
            else:
                #print("Không có kết quả")
                return ans, scs
    except Exception as e:
        # Xử lý ngoại lệ ở đây, ví dụ:
        print(f"Lỗi trong quá trình xử lý câu hỏi: {str(e)}")
        return [], []

def return_answer_aligned(question, upper_threshold, lower_threshold, embeddings_dataset):
    ans = []
    scs = []
    from sentence_transformers import util
    try:
        embed_question = embeddingvector_with_alignment(question).numpy()
        l2_score, samples = embeddings_dataset.get_nearest_examples(
            "embeddings", embed_question, k=3
        )
        #l2_similarity = [1 / (1 + distance) for distance in l2_score]
        cosine = [util.cos_sim(embed_question, i) for i in samples["embeddings"]]
        #print(l2_score)
        scores = [float(tensor.item()) for tensor in cosine]
        #print(l2_similarity)
        #scores = [0.2 * l2 + 0.8 * cosine for l2, cosine in zip(l2_similarity, scores)]

        result = [answer for answer, score in zip(samples['Trả lời'], scores) if score > upper_threshold]

        if len(result) >= 1:
            print("Câu trả lời chuẩn xác là:")
            ans.append(result[0])
            scs.append(scores[0])
            return ans, scs
        elif len(result) == 0:
            result = [answer for answer, score in zip(samples['Trả lời'], scores) if score >= lower_threshold]
            print(len(result))
            if len(result) > 0:
                for answer, score in zip(result, scores):
                    #print("Câu trả lời khả năng là:")
                    ans.append(answer)
                    scs.append(score)
                return ans, scs
            else:
                #print("Không có kết quả")
                return ans, scs
    except Exception as e:
        # Xử lý ngoại lệ ở đây, ví dụ:
        print(f"Lỗi trong quá trình xử lý câu hỏi: {str(e)}")
        return [], []


def return_answer(question, upper_threshold, lower_threshold, embeddings_dataset):
    ans = []
    scs = []
    from sentence_transformers import util
    try:
        embed_question = embeddingvector(question).numpy()
        l2_score, samples = embeddings_dataset.get_nearest_examples(
            "embeddings", embed_question, k=3
        )
        #l2_similarity = [1 / (1 + distance) for distance in l2_score]

        cosine = [util.cos_sim(embed_question, i) for i in samples["embeddings"]]
        #print(l2_score)
        scores = [float(tensor.item()) for tensor in cosine]
        #print(l2_similarity)
        #scores = [0.2 * l2 + 0.8 * cosine for l2, cosine in zip(l2_similarity, scores)]

        result = [answer for answer, score in zip(samples['Trả lời'], scores) if score > upper_threshold]

        if len(result) >= 1:
            print("Câu trả lời chuẩn xác là:")
            ans.append(result[0])
            scs.append(scores[0])
            return ans, scs
        elif len(result) == 0:
            result = [answer for answer, score in zip(samples['Trả lời'], scores) if score >= lower_threshold]
            print(len(result))
            if len(result) > 0:
                for answer, score in zip(result, scores):
                    #print("Câu trả lời khả năng là:")
                    ans.append(answer)
                    scs.append(score)
                return ans, scs
            else:
                #print("Không có kết quả")
                return ans, scs
    except Exception as e:
        # Xử lý ngoại lệ ở đây, ví dụ:
        print(f"Lỗi trong quá trình xử lý câu hỏi: {str(e)}")
        return [], []

# if __name__ == "__main__":
#     question = "Tôi có 10 triệu thì một ngày tôi được rút bao nhiêu tiền và mấy lần?"
#     upper_threshold = 0.9
#     lower_threshold = 0.3
#     a = []
#     b = []
#     a,b = return_answer(question, upper_threshold, lower_threshold)
#     # print(a)
#     # print(b)
#     for x, y in zip(a,b): 
#          print(x + ": " + str(y))
