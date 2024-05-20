# 读取 llava_bench 目录下的 tsv 文件
from email.mime import image
import os
import pandas as pd
from tqdm import tqdm
import io
import base64
from PIL import Image

# img_buffer = base64.b64encode(row['image_buffer_list'][0]['buffer']).decode('utf-8')

tsv_dir = "/home/lihaoyu/code/0516/llava_bench"
# 示例路径
tsv_ex_path = "/home/lihaoyu/code/0516/llava_bench/LLaVABench_French.tsv"

save_imgs_base_dir = "/home/lihaoyu/code/0516/llava_bench/imgs"

new_gpt_responses_base_dir = "/home/lihaoyu/code/0516/llava_bench/gpt_responses"

def read_base64_img_buffer(img_base64):
    img_byte_io = io.BytesIO(base64.b64decode(img_base64))
    img = Image.open(img_byte_io)
    return img

ori_tsv_list = [f for f in os.listdir(tsv_dir) if f.endswith(".tsv") and not f.endswith("_new.tsv")]
assert len(ori_tsv_list) == 4

for tsv_file in ori_tsv_list:
    lang = tsv_file.removesuffix(".tsv").split("_")[-1]
    tsv_path = os.path.join(tsv_dir, tsv_file)
    new_tsv_path = os.path.join(tsv_dir, tsv_file.removesuffix(".tsv") + "_new.tsv")

    df = pd.read_csv(tsv_path, sep='\t', index_col=0)

    print(f"reading tsv file {tsv_path}")
    # print(type(df))
    # print()

    # french: idx, question, image, image_path, category, gpt4_ans
    save_imgs_dir = os.path.join(save_imgs_base_dir, lang)
    if not os.path.exists(save_imgs_dir):
        os.mkdir(save_imgs_dir)

    # 下面这段注释掉的是从 tsv 中读取图片进行保存
    # question_list = df["question"].tolist()
    # img_base64_list = df["image"].tolist()
    # caption_list = df["caption"].tolist()
    # gpt4_ans_list = df["gpt4_ans"].tolist()
    # for idx in tqdm(range(len(img_base64_list))):
    #     question = question_list[idx]
    #     img_base64 = img_base64_list[idx]
    #     caption = caption_list[idx]
    #     gpt4_ans = gpt4_ans_list[idx]
        
    #     img_name = f"{lang}_{idx}.jpg"
    #     question_file_name = f"{lang}_{idx}.txt"
    #     caption_file_name = f"{lang}_{idx}_caption.txt"
    #     gpt4_ans_file_name = f"{lang}_{idx}_ori_gpt4_ans.txt"
        
    #     # img_path = os.path.join(save_imgs_dir, img_name)
    #     # img = read_base64_img_buffer(img_base64)
    #     # img.save(img_path)
        
    #     with open(os.path.join(save_imgs_dir, question_file_name), "w", encoding="utf-8") as fq:
    #         fq.write(question)
    #     with open(os.path.join(save_imgs_dir, caption_file_name), "w", encoding="utf-8") as fc:
    #         fc.write(caption)
    #     with open(os.path.join(save_imgs_dir, gpt4_ans_file_name), "w", encoding="utf-8") as fg:
    #         fg.write(gpt4_ans)
    
    # 修改 tsv 的 gpt_ans 列！
    for idx in tqdm(range(len(df))):
        # 路径示例：/home/lihaoyu/code/0516/llava_bench/gpt_responses/French/French_0_gpt4_ans.txt
        new_gpt_response_path = os.path.join(new_gpt_responses_base_dir, lang, f"{lang}_{idx}_gpt4_ans.txt")
        with open(new_gpt_response_path, "r", encoding="utf-8") as f:
            new_gpt_ans = f.read().strip()
            
        # 注意，葡萄牙语有三个例子问题是英文的，编号 34 56 59，需要从 /home/lihaoyu/code/0516/llava_bench/imgs/Portuguese/Portuguese_59.txt 等路径下手动读取
        if lang == "Portuguese" and idx in [34, 56, 59]:
            with open(f"/home/lihaoyu/code/0516/llava_bench/imgs/Portuguese/Portuguese_{idx}.txt", "r", encoding='utf-8') as f:
                translated_question = f.read().strip()
            df.loc[idx, 'question'] = translated_question
            
        df.loc[idx, 'gpt4_ans'] = new_gpt_ans
    
    # 将修改后的 tsv 保存到指定路径下
    df.to_csv(new_tsv_path, sep='\t', index=False, mode='w')
    
    

# df_ex = pd.read_csv(tsv_ex_path, sep="\t")
# img_buffer_ex = df_ex['image'].tolist()[0]
# print(type(img_buffer_ex))

# # img_ex = base64.b64encode(img_buffer_ex).decode('utf-8')
# img_byte_io = io.BytesIO(base64.b64decode(img_buffer_ex))
# img_ex = Image.open(img_byte_io)
# print(type(img_ex))

# img_ex.save("./example.jpg")