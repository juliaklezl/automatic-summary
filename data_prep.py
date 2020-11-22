#imports:
import re
import pandas as pd
from random import randint

def get_data(files):
    summaries = []
    texts = []
    for name in files:
        with open(name,'r', encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()
            text_num = 0 # text num starts counting from 0 for every new file
            s = ""
            e = 0
            for line in lines:
                strippedline = re.sub('[^0-9a-zA-Z. ]+', '', line) 
                if re.match('[0-9a-zA-Z]?abstract[0-9a-zA-Z ]*|abstract', strippedline): # add texts/summaries to the dfs
                    if s != "":
                        texts.append([name, text_num, s])
                    elif e != 0:
                        texts.append([name, text_num, e])
                    text_num += 1
                    s = ""  
                    e = 1  # e set to make sure empty lines are only added when abstract/text is empty otherwise (to keep counting even)
                elif re.match('[0-9a-zA-Z]?article[0-9a-zA-Z ]*|article', strippedline):
                    if s != "":
                        summaries.append([name, text_num, s])
                    elif e != 0:
                        summaries.append([name, text_num, e])
                    s = ""
                    e = 1
                elif (strippedline != "") and (strippedline != "\n") and (not re.match('^[0-9a-zA-Z.]* *$', strippedline)):
                    s = strippedline
                else:
                    if e == 1:
                        e = "empty"
            texts.append([name, text_num, s])
    texts_df = pd.DataFrame(texts, columns = ["filename", "text_num", "text"])
    summaries_df = pd.DataFrame(summaries, columns = ["filename", "text_num", "text"])
    return texts_df, summaries_df

def check_data(texts_df, summaries_df): # check whether every text in texts_df has a corresponding summary in summaries_df
    for i in range(0, len(summaries_df)):
        if texts_df["text_num"][i] != summaries_df["text_num"][i]:
            print("Problem:")
            print(texts_df["filename"][i])
            print(texts_df["text_num"][i])
            print(texts_df["text"][i])
            print(summaries_df["text_num"][i])
            print(summaries_df["text"][i])
    print("check complete")
    pass
                   
def random_sample(texts_df, summaries_df):
    i = randint(0, len(texts_df))
    print("file:", texts_df["filename"][i])
    print("article:", texts_df["text"][i])
    print("summary:", summaries_df["text"][i])
    return (texts_df["text"][i], summaries_df["text"][i])

