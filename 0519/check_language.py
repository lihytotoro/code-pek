# 检测输入的文本的语言
import pandas as pd
from tqdm import tqdm

# from langdetect import detect
# from langdetect import detect_langs
import fasttext
fasttext.FastText.eprint = lambda x: None

# df = pd.read_excel("./MiniCPM-V-2_LLaVABench_German.xlsx")
# df = pd.read_csv("../0516/llava_bench/LLaVABench_French_new.tsv", sep='\t', index_col=0)
df = pd.read_csv("../0516/llava_bench/LLaVABench_German_new.tsv", sep='\t', index_col=0)
# df = pd.read_csv("../0516/llava_bench/LLaVABench_Portuguese_new.tsv", sep='\t', index_col=0)
# df = pd.read_csv("../0516/llava_bench/LLaVABench_Spanish_new.tsv", sep='\t', index_col=0)

model = fasttext.load_model('lid.176.bin')

for row_idx, row in df.iterrows():
    quest = row['question'].replace("\n", "")
    gpt4_ans = row['gpt4_ans']
    
    quest_lang_predict = model.predict(quest, k=1)
    
    # print(quest_lang_predict[0][0])
    
    if quest_lang_predict[0][0] != '__label__de':
        print()
        print(f"question:{quest}")
        print(f"lang:{quest_lang_predict[0][0]}")
    
    # pred = row['prediction']
    
    # print(f"question:{detect(quest)}\t\tgpt4_ans:{detect(gpt4_ans)}")
    # print(f"question:{detect(quest)}\t\tprediction:{detect(pred)}")