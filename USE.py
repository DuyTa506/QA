import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import string
import tensorflow_hub as hub
import nltk
nltk.download('punkt')
nltk.download("wordnet")
from copy import deepcopy


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

with open("all_qa.json", "r", encoding='utf-8') as f:
    mobi_qa = json.load(f)

word_2_fullword = {"đk": "đăng ký",
                    "k/ko": "không",
                    "t": "tôi",
                    "CTV": "cộng tác viên",
                    "ctv": "cộng tác viên",
                    "lm": "làm",
                    "tk": "tài khoản",
                    "mk": "mật khẩu",
                    "ntn": "như thế nào",
                    "dt": "điện thoại",
                    "sđt/sdt": "số điện thoại",
                    "CCCD/cccd": "căn cước công dân",
                    "hđ/hd/hdong": "hoạt động",
                    "tn": "thế nào",
                    "DS/ds": "danh sách",
                    "yc": "yêu cầu",
                    "bn": "bao nhiêu",
                    "ls": "làm sao",
                    "đc/dc": "được",
                    "ktra": "kiểm tra"}
def bo_dau_thanh(ky_tu):
    chu_a_co_dau = ["à", "á", "ả", "ã", "ạ"]  # chữ 'a' với các dấu thanh
    chu_aw_co_dau = ["ằ", "ắ", "ẳ", "ẵ", "ặ"]  # chữ "ă" với các dấu thanh
    chu_aa_co_dau = ["ầ", "ấ", "ẩ", "ẫ", "ậ"]  # chữ 'â' với các dấu thanh
    chu_e_co_dau = ["è", "é", "ẻ", "ẽ", "ẹ"]  # chữ 'e' với các dấu thanh
    chu_ee_co_dau = ["ề", "ế", "ể", "ễ", "ệ"]  # chữ 'ê' với các dấu thanh
    chu_i_co_dau = ["ì", "í", "ỉ", "ĩ", "ị"]  # chữ 'i với các dấu thanh
    chu_o_co_dau = ["ò", "ó", "ỏ", "õ", "ọ"]  # chữ 'o' với các dấu thanh
    chu_oo_co_dau = ["ồ", "ố", "ổ", "ỗ", "ộ"]  # chữ "ô" với các dấu thanh
    chu_ow_co_dau = ["ờ", "ớ", "ở", "ỡ", "ợ"]  # chữ 'ơ' với các dấu thanh
    chu_u_co_dau = ["ù", "ú", "ủ", "ũ", "ụ"]  # chữ 'u' với các dấu thanh
    chu_uw_co_dau = ["ừ", "ứ", "ử", "ữ", "ự"]  # chữ 'ư' với các dấu thanh
    chu_y_co_dau = ["ỳ", "ý", "ỷ", "ỹ", "ỵ"]  # chữ 'y' với các dấu thanh

    chu_co_dau = [chu_a_co_dau, chu_aw_co_dau, chu_aa_co_dau,
                  chu_e_co_dau, chu_ee_co_dau, chu_i_co_dau,
                  chu_o_co_dau, chu_oo_co_dau, chu_ow_co_dau,
                  chu_u_co_dau, chu_uw_co_dau, chu_y_co_dau]

    chu_khong_dau = ['a', 'ă', 'â', 'e', 'ê', 'i',
                     'o', 'ô', 'ơ', 'u', 'ư', 'y']

    for index, ds_co_dau in enumerate(chu_co_dau):
        if ky_tu in ds_co_dau:
            return chu_khong_dau[index]
    return ky_tu



def preprocess_text(document):
    punctuation_removal = dict((ord(punctuation), None) for punctuation in string.punctuation)
    punctuation_removal_doc = document.lower().translate(punctuation_removal)
    tokens = nltk.word_tokenize(punctuation_removal_doc)
    wordnet_lemma = nltk.stem.WordNetLemmatizer()
    return [bo_dau_thanh(wordnet_lemma.lemmatize(word_2_fullword.get(token, token))) for token in tokens]

def generate_response(user_input, questions, question2answer):
    bot_response = ''
    data_sentences = []
    data_sentences.extend(questions)
    data_sentences.extend([user_input])
    all_word_vectors = embed(data_sentences)
    print(all_word_vectors.shape)
    print("all_word_vectors[-1]", all_word_vectors[-1].shape)
    similar_vector_values = cosine_similarity(
        all_word_vectors.numpy()[-1].reshape(1, -1), all_word_vectors)
    confident_score = np.sort(similar_vector_values)[0][-2]
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        bot_response = bot_response + "Xin lỗi tôi chưa hiểu câu hỏi của bạn"
        return bot_response
    else:
        similar_question = data_sentences[similar_sentence_number]
        bot_response = bot_response + question2answer[similar_question]
        return similar_question, confident_score, bot_response

question_list_accent = []
mobi_qa_coppy = {}
for question in mobi_qa.keys():
  new_question = preprocess_text(deepcopy(question))
  new_key = " ".join(new_question)
  mobi_qa_coppy[new_key] = mobi_qa[question]


continue_dialogue = True
print("Xin chào, tôi là trợ lý ảo, tôi sẽ trả lời câu hỏi của bạn:")
while(continue_dialogue == True):
    human_text = input()
    human_text = human_text.lower()
    human_text = " ".join(preprocess_text(human_text))
    print(human_text)
    if human_text != 'bye':
        if human_text == 'thanks' or human_text == 'thank you very much' or human_text == 'thank you':
            continue_dialogue = False
            print("AI Devup: Dạ em cám ơn ạ")
        else:
            print("AI Devup: " + str(generate_response(human_text, list(mobi_qa_coppy.keys()), mobi_qa_coppy)))
    else:
        continue_dialogue = False
        print("AI Devup: Tạm biệt và cám ơn quý khách...")