import argparse
import os.path as osp

import numpy as np
import pandas as pd

# 0519 new: fasttext
import fasttext
fasttext.FastText.eprint = lambda x: None

from vlmeval.evaluate.misc import build_judge
from vlmeval.smp import defaultdict, dump, get_logger, load
from vlmeval.utils import track_progress_rich

rule_dict = {
    'llava_bench_conv': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'},  # noqa: E501
    'llava_bench_detail': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'},  # noqa: E501
    'llava_bench_complex': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'}  # noqa: E501
}


def get_eval(judge, content):
    return judge.generate(content)


def parse_score(review):
    logger = get_logger('Evaluation')
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            logger.error('error', review)
            return [-1, -1]
    except Exception as e:
        logger.error(e, 'error', review)
        return [-1, -1]


def build_prompt(line):
    cap_str = line['caption']
    question = line['question']
    ans1 = line['gpt4_ans']
    ans2 = line['prediction']
    category = 'llava_bench_' + line['category']
    rule = rule_dict[category]
    role, prompt = rule['role'], rule['prompt']

    content = (f'[Context]\n{cap_str}\n\n'
               f'[Question]\n{question}\n\n'
               f'[{role} 1]\n{ans1}\n\n[End of {role} 1]\n\n'
               f'[{role} 2]\n{ans2}\n\n[End of {role} 2]\n\n'
               f'[System]\n{prompt}\n\n')
    # 0519 new: change return value to tuple in order to include line(question and prediction)
    return (content, (question, ans2))


def LLaVABench_atomeval(model, prompt):
    '''
    prompt: (content, (question, ans2))
    '''
    # 参数加上lines
    # 然后score再加上一个纬度，是是否语种相同
    review = get_eval(model[0], prompt[0])
    scores = parse_score(review)
    
    # 0519 new: judge the language type consisitency
    question = prompt[1][0]
    prediction = prompt[1][1]
    fasttext_model = model[1]
    
    # 给 scores 新增加一个维度
    if check_language_consistency(question, prediction):
        scores.append(1)
    else:
        scores.append(0)
    
    return scores


def LLaVABench_score(data):
    cates = ['overall'] + list(set(data['category']))
    ret = defaultdict(list)

    for c in cates:
        ret['split'].append(c)
        sub = data[data['category'] == c] if c != 'overall' else data
        ret['Relative Score (main)'].append(np.mean(sub['score']) / np.mean(sub['gpt4_score']) * 100)
        ret['VLM Score'].append(np.mean(sub['score']) * 10)
        ret['GPT4 Score'].append(np.mean(sub['gpt4_score']) * 10)
    return pd.DataFrame(ret)


def LLaVABench_eval(eval_file, **judge_kwargs):
    suffix = '.' + eval_file.split('.')[-1]
    record_file = eval_file.replace(suffix, '_openai_result' + suffix)
    score_file = eval_file.replace(suffix, '_score.csv')
    nproc = judge_kwargs.pop('nproc', 4)

    # 0519 new: load fasttext model
    fasttext_model_path = judge_kwargs["fasttext_model_path"]
    fasttext_model = fasttext.load_model(fasttext_model_path)

    if not osp.exists(record_file):
        data = load(eval_file)
        lines = [data.iloc[i] for i in range(len(data))]
        model = build_judge(
            temperature=0.2,
            system_prompt='You are a helpful and precise assistant for checking the quality of the answer.',
            **judge_kwargs)
        # 0519 new: now prompt is a tuple! (content, (question, ans2))
        # also, tups need to include fasttext model!
        prompts = [build_prompt(line) for line in lines]
        tups = [((model, fasttext_model), prompt) for prompt in prompts]
        
        scores = track_progress_rich(LLaVABench_atomeval, tups, nproc=nproc, chunksize=nproc)
        data['gpt4_score'] = [x[0] for x in scores]
        # data['score'] = [x[1] for x in scores] # 这里 if 语种相同直接用 x[1], 不同 = 1
        data_score = []
        for x in scores:
            if x[2] == 1:
                data_score.append(x[1])
            else:
                data_score.append(1.)
        data['score'] = data_score
        
        dump(data, record_file)

    data = load(record_file)
    ret = LLaVABench_score(data).round(1)
    print(ret)
    dump(ret, score_file)
    return ret


# 0519 new: 检查两段文本的语种是否一致
def check_language_consistency(text1, text2, fasttext_model):
    lang1 = fasttext_model.predict(text1, k=1)[0][0]
    lang2 = fasttext_model.predict(text2, k=1)[0][0]
    
    return lang1 == lang2


def parse_args():
    parser = argparse.ArgumentParser(description='LLaVABench Evaluation. ')
    parser.add_argument('data', type=str, help='The question set for inference, in excel / tsv / json format. ')
    parser.add_argument(
        '--model', type=str, help='The LLM (GPT) used for inference. ', default='gpt-4-turbo',
        choices=['gpt-4-0613', 'gpt-4-turbo', 'chatgpt-1106', 'chatgpt-0613', 'gpt-4-0314'])
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')
    
    # 0519 new: fasttext model args
    parser.add_argument("--fasttext_model_path", type=str, default="/home/lihaoyu/code/0519/lid.176.bin", help="")
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    LLaVABench_eval(eval_file=args.data, model=args.model, nproc=args.nproc, verbose=args.verbose, fasttext_model_path=args.fasttext_model_path)
