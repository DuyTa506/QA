
from pyvi import ViUtils
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from trigram import trigram
#corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction")
# token  = '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'

# PhobertTokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
# model = AutoModelForSeq2SeqLM.from_pretrained("bmd1905/vietnamese-correction")
def check_spell(text):
    word_dict = {"đk": "đăng ký",
                    "k/ko": "không",
                    "t": "tôi",
                    "lm": "làm",
                    "tk": "tài khoản",
                    "mk": "mật khẩu",
                    "ntn": "như thế nào",
                    "dt": "điện thoại",
                    "sđt/sdt": "số điện thoại",
                    "cccd": "căn cước công dân",
                    "hđ/hd/hdong": "hoạt động",
                    "tn": "thế nào",
                    "ds/đs": "danh sách",
                    "yc": "yêu cầu",
                    "bn": "bao nhiêu",
                    "ls": "làm sao",
                    "đc/dc": "được",
                    "ktra": "kiểm tra",
                    "ll": "liên lạc",
                    "lh": "liên hệ"}
    words = text.split()
    corrected_text = []

    i = 0
    while i < len(words):
        word = words[i]
        # Kiểm tra xem từ có '/' không
        if '/' in word:
            options = word.split('/')
            found = False
            for option in options:
                if option in word_dict:
                    corrected_text.append(word_dict[option])
                    found = True
                    break
            if not found:
                corrected_text.append(word)
        else:
            found = False
            # Kiểm tra xem từ có trong danh sách các từ viết tắt không
            for key in word_dict:
                if word in key.split('/'):
                    
                    corrected_text.append(word_dict[key])
                    found = True
                    break
            if not found:
                corrected_text.append(word)
        i += 1
        out = ' '.join(corrected_text)
    try:
            # Thử sử dụng ViUtils để chuyển đổi
            out = trigram(out).lower()
    except Exception as e:
            return out

    return out
# question = "khi muon doi so dien thoai thi co cah nao"
# align = check_spell((question))

# inputs = PhobertTokenizer(
#         align, padding=True, truncation=True, return_tensors="pt")['input_ids']
# with torch.no_grad():
#         model_output = model.generate(inputs)
# print(model_output)
# end_correct= PhobertTokenizer.batch_decode(model_output, skip_special_tokens=True)

# print(end_correct)

