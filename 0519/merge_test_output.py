# 读取所有与 llava_bench 相关的结果，读到一个 csv 里
import pandas as pd
import os

# target_dir = "/home/lihaoyu/code/0519/Yi_VL_34B"
target_dir = "/home/cuijunbo/github_repo/0427/VLMEvalKit/minicpmv_DPO-minicpmv_llama3_multilingual_1iter_greedy_sr4000img_bs1_gradacc4_beta0.3_lr5e-7_fp32-minicpmv_llama3_multilingual_1iter_greedy_sr4000img-1_checkpoint-280"

# target_files = [f for f in os.listdir(target_dir) if f .endswith(".csv") and f.startswith("Yi") and "LLaVABench" in f]
target_files = [f for f in os.listdir(target_dir) if f .endswith(".csv") and f.startswith("minicpmv") and "LLaVABench" in f]

# 15
print(len(target_files))

task_list, score_list = [], []

for f in target_files:
    # 获取任务相关的名称
    # prefix = "Yi_VL_34B_LLaVABench_"
    prefix = "minicpmv_DPO-minicpmv_llama3_multilingual_1iter_greedy_sr4000img_bs1_gradacc4_beta0.3_lr5e-7_fp32-minicpmv_llama3_multilingual_1iter_greedy_sr4000img-1_checkpoint-280_LLaVABench_"
    task_name = f.removeprefix(prefix).removesuffix("_score.csv")
    task_list.append(task_name)
    
    target_path = os.path.join(target_dir, f)
    df = pd.read_csv(target_path, encoding='utf-8')
    
    df_overall = df[df['split'] == 'overall']
    assert len(df_overall) == 1
    
    score = df_overall.iloc[0]['Relative Score (main)']
    score_list.append(float(score))
    
df_merge = pd.DataFrame({'task':task_list, 'overall':score_list})
print(df_merge)

df_merge.to_csv("./merged_llavabench_score_1.csv", encoding='utf-8', index=False)