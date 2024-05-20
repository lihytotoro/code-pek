# 整合目录下所有以 openai_results 结尾的 xlsx，使用 fasttext 规则，计算每个 xlsx (task) 的得分，最后整理成 csv
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import fasttext
fasttext.FastText.eprint = lambda x: None

# target_dir = "/home/cuijunbo/github_repo/0427/VLMEvalKit/Yi_VL_34B"
# target_dir = "/home/cuijunbo/github_repo/0427/VLMEvalKit/minicpmv_DPO-minicpmv_llama3_multilingual_1iter_greedy_sr4000img_bs1_gradacc4_beta0.3_lr5e-7_fp32-minicpmv_llama3_multilingual_1iter_greedy_sr4000img-1_checkpoint-280"

target_dir = "/home/lihaoyu/code/0519/Yi_VL_34B"
# target_dir = "/home/lihaoyu/code/0519/minicpmv_DPO-minicpmv_llama3_multilingual_1iter_greedy_sr4000img_bs1_gradacc4_beta0.3_lr5e-7_fp32-minicpmv_llama3_multilingual_1iter_greedy_sr4000img-1_checkpoint-280"

target_files = [f for f in os.listdir(target_dir) if f.endswith("_openai_result.xlsx") and f.startswith("Yi") and "LLaVABench" in f]
# target_files = [f for f in os.listdir(target_dir) if f.endswith("_openai_result.xlsx") and "LLaVABench" in f]
# for f in target_files:
    # print(f)
# assert len(target_files) == 15
print(len(target_files))


def check_language_consistency(text1, text2, fasttext_model):
    text11 = text1.replace("\n", "")
    text22 = text2.replace("\n", "")
    
    lang1 = fasttext_model.predict(text11, k=1)[0][0]
    lang2 = fasttext_model.predict(text22, k=1)[0][0]
    
    return lang1 == lang2


if __name__ == "__main__":
    # 读取 fasttext model，用于检测 question 和 prediction 语言是否同类！
    fasttext_model_path = '/home/lihaoyu/code/0519/lid.176.bin'
    fasttext_model = fasttext.load_model(fasttext_model_path)
    
    task_list, total_gpt4_score_list, total_score_list, relative_score_list = [], [], [], []
    
    for f in target_files:
        prefix = "Yi_VL_34B_LLaVABench_"
        # prefix = "minicpmv_DPO-minicpmv_llama3_multilingual_1iter_greedy_sr4000img_bs1_gradacc4_beta0.3_lr5e-7_fp32-minicpmv_llama3_multilingual_1iter_greedy_sr4000img-1_checkpoint-280_LLaVABench_"
        suffix = "_openai_result.xlsx"
        task_name = f.removeprefix(prefix).removesuffix(suffix)
        task_list.append(task_name)
        
        target_path = os.path.join(target_dir, f)
        df = pd.read_excel(target_path, index_col=0)
        
        # 对于每个 xlsx 文件，遍历 df，修改某些位置的 score，然后计算平均值？
        for row_idx, row in df.iterrows():
            question = row['question']
            prediction = row['prediction']
            
            # 对所有不符合要求的行，将 score 列修改为 1！
            if not check_language_consistency(question, prediction, fasttext_model):
                df.loc[row_idx, 'score'] = 1
                
        total_gpt4_score = np.sum(np.array(df['gpt4_score'].tolist()))
        total_score = np.sum(np.array(df['score'].tolist()))
    
        relative_score = float(total_score) / float(total_gpt4_score) * 100
        
        total_gpt4_score_list.append(total_gpt4_score)
        total_score_list.append(total_score)
        relative_score_list.append(relative_score)
        
    #
    df_merge = pd.DataFrame({'task':task_list, 'total_score':total_score_list, 'total_gpt4_score':total_gpt4_score_list, \
                        'relative_score':relative_score_list})
    print(df_merge)
    
    # df_merge.to_csv("./outputs/merged_llavabench_relative_score_minicpmv.csv", encoding='utf-8', index=False)
    df_merge.to_csv("./outputs/merged_llavabench_relative_score_YiVL.csv", encoding='utf-8', index=False)