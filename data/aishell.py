import os
import wget
import tarfile
import argparse
import subprocess
from utils import create_manifest
from tqdm import tqdm
import shutil
import re
import string
import unicodedata

root = "Aishell_dataset/"

def traverse(root, path, search_fix=".txt"):
    f_list = []

    p = root + path
    for s_p in sorted(os.listdir(p)):
        for sub_p in sorted(os.listdir(p + "/" + s_p)):
            if sub_p[len(sub_p)-len(search_fix):] == search_fix:
                print(">", path, s_p, sub_p)
                f_list.append(p + "/" + s_p + "/" + sub_p)

    return f_list

def remove_punctuation(seq):
    # REMOVE CHINESE PUNCTUATION EXCEPT HYPEN / DASH AND FULL STOP
    seq = re.sub("[\s+\\!\/_,$%=^*?:@&^~`(+\"]+|[+！，。？、~@#￥%……&*（）:;：；《）《》“”()»〔〕]+", " ", seq)
    seq = seq.replace(" ' ", " ")
    seq = seq.replace(" ’ ", " ")
    seq = seq.replace(" ＇ ", " ")
    seq = seq.replace(" ` ", " ")

    seq = seq.replace(" '", "'")
    seq = seq.replace(" ’", "’")
    seq = seq.replace(" ＇", "＇")

    seq = seq.replace("’ ", " ")
    seq = seq.replace("＇ ", " ")
    seq = seq.replace("` ", " ")
    seq = seq.replace(".", "")

    seq = seq.replace("`", "")
    seq = seq.replace("-", " ")
    seq = seq.replace("?", " ")
    seq = seq.replace(":", " ")
    seq = seq.replace(";", " ")
    seq = seq.replace("]", " ")
    seq = seq.replace("[", " ")
    seq = seq.replace("}", " ")
    seq = seq.replace("{", " ")
    seq = seq.replace("|", " ")
    seq = seq.replace("_", " ")
    seq = seq.replace("(", " ")
    seq = seq.replace(")", " ")
    seq = seq.replace("=", " ")

    seq = seq.replace("doens't", "doesn't")
    seq = seq.replace("o' clock", "o'clock")
    seq = seq.replace("因为it's", "因为 it's")
    seq = seq.replace("it' s", "it's")
    seq = seq.replace("it ' s", "it's")
    seq = seq.replace("it' s", "it's")
    seq = seq.replace("y'", "y")
    seq = seq.replace("y ' ", "y")
    seq = seq.replace("看different", "看 different")
    seq = seq.replace("it'self", "itself")
    seq = seq.replace("it'ss", "it's")
    seq = seq.replace("don'r", "don't")
    seq = seq.replace("has't", "hasn't")
    seq = seq.replace("don'know", "don't know")
    seq = seq.replace("i'll", "i will")
    seq = seq.replace("you're", "you are")
    seq = seq.replace("'re ", " are ")
    seq = seq.replace("'ll ", " will ")
    seq = seq.replace("'ve ", " have ")
    seq = seq.replace("'re\n", " are\n")
    seq = seq.replace("'ll\n", " will\n")
    seq = seq.replace("'ve\n", " have\n")

    seq = remove_space_in_between_words(seq)
    return seq

def remove_special_char(seq):
    seq = re.sub("[【】·．％°℃×→①ぃγ￣σς＝～•＋δ≤∶／⊥＿ñãíå∈△β［］±]+", " ", seq)
    return seq

def remove_space_in_between_words(seq):
    return seq.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").strip().lstrip()

def remove_return(seq):
    return seq.replace("\n", "").replace("\r", "").replace("\t", "")

def preprocess(seq):
    seq = seq.lower()
    seq = re.sub("[\(\[].*?[\)\]]", "", seq) # REMOVE ALL WORDS WITH BRACKETS (HESITATION)
    seq = re.sub("[\{\[].*?[\}\]]", "", seq) # REMOVE ALL WORDS WITH BRACKETS (HESITATION)
    seq = re.sub("[\<\[].*?[\>\]]", "", seq) # REMOVE ALL WORDS WITH BRACKETS (HESITATION)
    seq = re.sub("[\【\[].*?[\】\]]", "", seq) # REMOVE ALL WORDS WITH BRACKETS (HESITATION)
    seq = seq.replace("\x7f", "")
    seq = seq.replace("\x80", "")
    seq = seq.replace("\u3000", " ")
    seq = seq.replace("\xa0", "")
    seq = seq.replace("[", " [")
    seq = seq.replace("]", "] ")
    seq = seq.replace("#", "")
    seq = seq.replace(",", "")
    seq = seq.replace("*", "")
    seq = seq.replace("\n", "")
    seq = seq.replace("\r", "")
    seq = seq.replace("\t", "")
    seq = seq.replace("~", "")
    seq = seq.replace("—", "")
    seq = seq.replace("  ", " ").replace("  ", " ")
    seq = re.sub('\<.*?\>','', seq) # REMOVE < >
    seq = re.sub('\【.*?\】','', seq) # REMOVE 【 】
    seq = remove_special_char(seq)
    seq = remove_space_in_between_words(seq)
    seq = seq.strip()
    seq = seq.lstrip()
    seq = remove_punctuation(seq)
    seq = remove_space_in_between_words(seq)

    return seq

def is_chinese_char(cc):
    return unicodedata.category(cc) == 'Lo'

def is_contain_chinese_word(seq):
    for i in range(len(seq)):
        if is_chinese_char(seq[i]):
            return True
    return False

CHINESE_TAG = "†"
ENGLISH_TAG = "‡"

def add_lang(seq):
    new_seq = ""
    words = seq.split(" ")
    lang = 0
    for i in range(len(words)):
        if is_contain_chinese_word(words[i]):
            if lang != 1:
                lang = 1
                new_seq += CHINESE_TAG
                # print("zh")
        else:
            if lang != 2:
                lang = 2
                new_seq += ENGLISH_TAG
                # print("en")
        if new_seq != "":
            new_seq += " "
        new_seq += words[i]
    return new_seq

def separate_chinese_chars(seq):
    new_seq = ""
    words = seq.split(" ")
    for i in range(len(words)):
        if is_contain_chinese_word(words[i]):
            for char in words[i]:
                if new_seq != "":
                    new_seq += " "
                new_seq += char
        else:
            if new_seq != "":
                new_seq += " "
            new_seq += words[i]
    return new_seq

print("PREPROCESSING")
if not os.path.isdir("Aishell_dataset/transcript_clean"):
    os.system("mkdir Aishell_dataset/transcript_clean")
    os.system("mkdir Aishell_dataset/transcript_clean/train")
    os.system("mkdir Aishell_dataset/transcript_clean/dev")
    os.system("mkdir Aishell_dataset/transcript_clean/test")

if not os.path.isdir("Aishell_dataset/transcript_clean_lang"):
    os.system("mkdir Aishell_dataset/transcript_clean_lang")
    os.system("mkdir Aishell_dataset/transcript_clean_lang/train")
    os.system("mkdir Aishell_dataset/transcript_clean_lang/dev")
    os.system("mkdir Aishell_dataset/transcript_clean_lang/test")

# PREPROCESS (additional remove all punctuation except '), and all hesitations
tr_file_list = traverse(root, "transcript/train", search_fix="")
dev_file_list = traverse(root, "transcript/dev", search_fix="")
test_file_list = traverse(root, "transcript/test", search_fix="")

for i in range(len(tr_file_list)):
    text_file_path = tr_file_list[i]
    new_text_file_path = tr_file_list[i].replace("transcript", "transcript_clean").replace(".wav","") 
    new_text_file_lang_path = tr_file_list[i].replace("transcript", "transcript_clean_lang").replace(".wav","") 
    print(new_text_file_path)
    
    with open(text_file_path, "r", encoding="utf-8") as text_file:
        for line in text_file:
            print(line)
            line = preprocess(line).rstrip().strip().lstrip()
            print(">", line)
            lang_line = add_lang(line)
            lang_line = separate_chinese_chars(lang_line).replace("  ", " ")
            print(">>", lang_line)

            if len(line) > 0:
                if not os.path.isdir(os.path.dirname(new_text_file_path)):
                    os.makedirs(os.path.dirname(new_text_file_path))
                with open(new_text_file_path, "w+", encoding="utf-8") as new_text_file:
                    new_text_file.write(line + "\n")

                if not os.path.isdir(os.path.dirname(new_text_file_lang_path)):
                    os.makedirs(os.path.dirname(new_text_file_lang_path))
                with open(new_text_file_lang_path, "w+", encoding="utf-8") as new_text_lang_file:
                    new_text_lang_file.write(lang_line + "\n")
            break
            
for i in range(len(dev_file_list)):
    text_file_path = dev_file_list[i]
    new_text_file_path = dev_file_list[i].replace("transcript", "transcript_clean").replace(".wav",".txt") 
    new_text_file_lang_path = dev_file_list[i].replace("transcript", "transcript_clean_lang").replace(".wav","") 
    print(new_text_file_path)
    
    with open(text_file_path, "r", encoding="utf-8") as text_file:
        for line in text_file:
            print(line)
            line = preprocess(line).rstrip().strip().lstrip()
            print(">", line)
            lang_line = add_lang(line)
            lang_line = separate_chinese_chars(lang_line).replace("  ", " ")
            print(">>", lang_line)

            if len(line) > 0:
                if not os.path.isdir(os.path.dirname(new_text_file_path)):
                    os.makedirs(os.path.dirname(new_text_file_path))
                with open(new_text_file_path, "w+", encoding="utf-8") as new_text_file:
                    new_text_file.write(line + "\n")

                if not os.path.isdir(os.path.dirname(new_text_file_lang_path)):
                    os.makedirs(os.path.dirname(new_text_file_lang_path))
                with open(new_text_file_lang_path, "w+", encoding="utf-8") as new_text_lang_file:
                    new_text_lang_file.write(lang_line + "\n")
            break
            
for i in range(len(test_file_list)):
    text_file_path = test_file_list[i]
    new_text_file_path = test_file_list[i].replace("transcript", "transcript_clean").replace(".wav",".txt") 
    new_text_file_lang_path = test_file_list[i].replace("transcript", "transcript_clean_lang").replace(".wav","") 
    print(new_text_file_path)
    
    with open(text_file_path, "r", encoding="utf-8") as text_file:
        for line in text_file:
            print(line)
            line = preprocess(line).rstrip().strip().lstrip()
            print(">", line)
            lang_line = add_lang(line)
            lang_line = separate_chinese_chars(lang_line).replace("  ", " ")
            print(">>", lang_line)

            if len(line) > 0:
                if not os.path.isdir(os.path.dirname(new_text_file_path)):
                    os.makedirs(os.path.dirname(new_text_file_path))
                with open(new_text_file_path, "w+", encoding="utf-8") as new_text_file:
                    new_text_file.write(line + "\n")

                if not os.path.isdir(os.path.dirname(new_text_file_lang_path)):
                    os.makedirs(os.path.dirname(new_text_file_lang_path))
                with open(new_text_file_lang_path, "w+", encoding="utf-8") as new_text_lang_file:
                    new_text_lang_file.write(lang_line + "\n")
            break

tr_file_list = traverse(root, "wav/train", search_fix="")
dev_file_list = traverse(root, "wav/dev", search_fix="")
test_file_list = traverse(root, "wav/test", search_fix="")

print("MANIFEST")
print(">>", len(tr_file_list))
print(">>", len(dev_file_list))
print(">>", len(test_file_list))

labels = {}
labels["_"] = True

alpha = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
labels = {}
labels["_"] = True
for char in alpha:
    labels[char] = True

with open("manifests/aishell_train_manifest.csv", "w+") as train_manifest:
    for i in range(len(tr_file_list)):
        wav_filename = tr_file_list[i]
        text_filename = tr_file_list[i].replace(".wav", ".txt").replace("wav", "transcript_clean")

        if os.path.isfile(text_filename):
            print(text_filename)
            with open(text_filename, "r", encoding="utf-8") as trans_file:
                for line in trans_file:
                    for char in line:
                        if char != "\n" and char != "\r" and char != "\t":
                            labels[char] = True

            train_manifest.write("data/" + wav_filename + "," + "data/" + text_filename + "\n")

with open("manifests/aishell_dev_manifest.csv", "w+") as valid_manifest:
    for i in range(len(dev_file_list)):
        wav_filename = dev_file_list[i]
        text_filename = dev_file_list[i].replace(".wav", ".txt").replace("wav", "transcript_clean")

        if os.path.isfile(text_filename):
            print(text_filename)
            with open(text_filename, "r", encoding="utf-8") as trans_file:
                for line in trans_file:
                    for char in line:
                        if char != "\n" and char != "\r" and char != "\t":
                            labels[char] = True

            valid_manifest.write("data/" + wav_filename + "," + "data/" + text_filename + "\n")

with open("manifests/aishell_test_manifest.csv", "w+") as test_manifest:
    for i in range(len(test_file_list)):
        wav_filename = test_file_list[i]
        text_filename = test_file_list[i].replace(".wav", ".txt").replace("wav", "transcript_clean")

        if os.path.isfile(text_filename):
            print(text_filename)
            with open(text_filename, "r", encoding="utf-8") as trans_file:
                for line in trans_file:
                    for char in line:
                        if char != "\n" and char != "\r" and char != "\t":
                            labels[char] = True

            test_manifest.write("data/" + wav_filename + "," + "data/" + text_filename + "\n")

with open("labels/aishell_labels.json", "w+") as labels_json:
    labels_json.write("[")
    i = 0
    labels_json.write('\n"_"')
    for char in labels:
        if char == "" or char == "_" or char == " ":
            continue
        labels_json.write(',\n')
        
        if char == "\\":
            print("slash")
            labels_json.write('"')
            labels_json.write('\\')
            labels_json.write('\\')
            labels_json.write('"')
        elif char == '"':
            print('double quote', i, char)
            labels_json.write('"\\""')
        else:
            labels_json.write('"' + char + '"')

        i += 1
    labels_json.write(',\n" "\n]')

print(len(labels))

### WITH LANG ###

with open("manifests/aishell_train_lang_manifest.csv", "w+") as train_manifest:
    for i in range(len(tr_file_list)):
        wav_filename = tr_file_list[i]
        text_filename = tr_file_list[i].replace(".wav", ".txt").replace("wav", "transcript_clean_lang")

        if os.path.isfile(text_filename):
            print(text_filename)
            with open(text_filename, "r", encoding="utf-8") as trans_file:
                for line in trans_file:
                    for char in line:
                        if char != "\n" and char != "\r" and char != "\t":
                            labels[char] = True

            train_manifest.write("data/" + wav_filename + "," + "data/" + text_filename + "\n")

with open("manifests/aishell_dev_lang_manifest.csv", "w+") as valid_manifest:
    for i in range(len(dev_file_list)):
        wav_filename = dev_file_list[i]
        text_filename = dev_file_list[i].replace(".wav", ".txt").replace("wav", "transcript_clean_lang")

        if os.path.isfile(text_filename):
            print(text_filename)
            with open(text_filename, "r", encoding="utf-8") as trans_file:
                for line in trans_file:
                    for char in line:
                        if char != "\n" and char != "\r" and char != "\t":
                            labels[char] = True

            valid_manifest.write("data/" + wav_filename + "," + "data/" + text_filename + "\n")

with open("manifests/aishell_test_lang_manifest.csv", "w+") as test_manifest:
    for i in range(len(test_file_list)):
        wav_filename = test_file_list[i]
        text_filename = test_file_list[i].replace(".wav", ".txt").replace("wav", "transcript_clean_lang")

        if os.path.isfile(text_filename):
            print(text_filename)
            with open(text_filename, "r", encoding="utf-8") as trans_file:
                for line in trans_file:
                    for char in line:
                        if char != "\n" and char != "\r" and char != "\t":
                            labels[char] = True

            test_manifest.write("data/" + wav_filename + "," + "data/" + text_filename + "\n")

with open("labels/aishell_lang_labels.json", "w+") as labels_json:
    labels_json.write("[")
    i = 0
    labels_json.write('\n"_"')
    labels[CHINESE_TAG] = True
    labels[ENGLISH_TAG] = True
    for char in labels:
        if char == "" or char == "_" or char == " ":
            continue
        labels_json.write(',\n')
        
        if char == "\\":
            print("slash")
            labels_json.write('"')
            labels_json.write('\\')
            labels_json.write('\\')
            labels_json.write('"')
        elif char == '"':
            print('double quote', i, char)
            labels_json.write('"\\""')
        else:
            labels_json.write('"' + char + '"')

        i += 1
    labels_json.write(',\n" "\n]')

print(len(labels))