from nltk import ngrams
from collections import defaultdict
from underthesea import word_tokenize
from tqdm import tqdm
import pickle
from os import path

import numpy as np
import re

def _zero():
  return 0

def _ngram():
  return defaultdict(_zero)

# Load language model
with open('trigrams/bi_letter.plk', 'rb') as fout:
    bigram_letter = pickle.load(fout)

# total word
total_word = 0
for k in bigram_letter.keys():
    total_word += sum(bigram_letter[k].values())

#
vocab_size = len(bigram_letter)

# tính xác suất dùng smoothing
def get_proba(current_word, next_word):
    if current_word not in bigram_letter:
        return 1 / total_word
    if next_word not in bigram_letter[current_word]:

        return 1 / (sum(bigram_letter[current_word].values()) + vocab_size)
    return (bigram_letter[current_word][next_word] + 1) / (sum(bigram_letter[current_word].values()) + vocab_size)


def remove_vn_accent(word):
    word = re.sub('[áàảãạăắằẳẵặâấầẩẫậ]', 'a', word)
    word = re.sub('[éèẻẽẹêếềểễệ]', 'e', word)
    word = re.sub('[óòỏõọôốồổỗộơớờởỡợ]', 'o', word)
    word = re.sub('[íìỉĩị]', 'i', word)
    word = re.sub('[úùủũụưứừửữự]', 'u', word)
    word = re.sub('[ýỳỷỹỵ]', 'y', word)
    word = re.sub('đ', 'd', word)
    return word


def gen_accents_word(word):
    word_no_accent = remove_vn_accent(word.lower())
    all_accent_word = {word}
    for w in open('trigrams/vn_syllables.txt', encoding="utf8").read().splitlines():
        w_no_accent = remove_vn_accent(w.lower())
        if w_no_accent == word_no_accent:
            all_accent_word.add(w)
    return all_accent_word

# hàm beam search


def beam_search(words, k=1):
  sequences = []
  for idx, word in enumerate(words):
    if idx == 0:
      sequences = [([x], 0.0) for x in gen_accents_word(word)]
    else:
      all_sequences = []
      for seq in sequences:
        for next_word in gen_accents_word(word):
          print(next_word)
          current_word = seq[0][-1]
          print(current_word)
          proba = get_proba(current_word, next_word)
          # print(current_word, next_word, proba, log(proba))
          proba = np.log(proba)
          new_seq = seq[0].copy()
          new_seq.append(next_word)
          all_sequences.append((new_seq, seq[1] + proba))
      # sắp xếp và lấy k kết quả ngon nhất
      all_sequences = sorted(all_sequences, key=lambda x: x[1], reverse=True)
      sequences = all_sequences[:k]
  return sequences
def greedy_search(words):
  sequences = []
  for idx, word in enumerate(words):
    if idx == 0:
      sequences = [([x], 0.0) for x in gen_accents_word(word)]
    else:
      all_sequences = []
      for seq in sequences:
        for next_word in gen_accents_word(word):
          current_word = seq[0][-1]
          proba = get_proba(current_word, next_word)
          # print(current_word, next_word, proba, log(proba))
          proba = np.log(proba)
          new_seq = seq[0].copy()
          new_seq.append(next_word)
          all_sequences.append((new_seq, seq[1] + proba))

    for row in data:
        # Select the index of the word with the highest probability at each step
        best_index = row.index(max(row))
        sequence.append(best_index)

    return sequence




from nltk.tokenize.treebank import TreebankWordDetokenizer
def trigram(_input):

    detokenize = TreebankWordDetokenizer().detokenize

    result = beam_search(_input.lower().split())

    return detokenize(result[0][0])

print(trigram("Co bat ky quy dinh cu the nao ve viec mat phi khi rut hoa hong vao tai khoan ngan hang khong"))