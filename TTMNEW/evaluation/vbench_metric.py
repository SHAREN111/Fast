from vbench import VBench
import sys
import json
import os
import os.path as osp
from tqdm import tqdm
import time
import numpy as np
import torch
import cv2
from glob import glob
import argparse
from PIL import Image
from visual_metric import cal_vis_metric
sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cpu_num = 2
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.multiprocessing.set_sharing_strategy('file_system')

from constant import (
    TASK_INFO,
    NORMALIZE_DIC,
    DIM_WEIGHT,
    QUALITY_LIST,
    SEMANTIC_LIST,
    QUALITY_WEIGHT,
    SEMANTIC_WEIGHT
)

# -----------------------
#   VBench 官方 total-score 计算函数
# -----------------------

def get_normalized_score(upload_data):
    normalized_score = {}
    for key in TASK_INFO:
        min_val = NORMALIZE_DIC[key]['Min']
        max_val = NORMALIZE_DIC[key]['Max']
        norm = (upload_data[key] - min_val) / (max_val - min_val)
        normalized_score[key] = norm * DIM_WEIGHT[key]
    return normalized_score

def get_quality_score(normalized_score):
    s = sum(normalized_score[key] for key in QUALITY_LIST)
    w = sum(DIM_WEIGHT[key] for key in QUALITY_LIST)
    return s / w

def get_semantic_score(normalized_score):
    s = sum(normalized_score[key] for key in SEMANTIC_LIST)
    w = sum(DIM_WEIGHT[key] for key in SEMANTIC_LIST)
    return s / w

def get_final_score(quality_score, semantic_score):
    return (quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT) / \
           (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

def cal_score(videos_path, result_dir, name='all', unpruned_videos_path='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/dongchengqi/InfinityStar/TTM-dev/evaluation/raw', time=0, memory=0):
    # mode = os.path.basename(videos_path)
    # name = "all"
    # result_dir = f"/data3/chengqidong/mrg/InfinityStar/evaluation/results/{mode}"
    # os.makedirs(result_dir, exist_ok=True)
    PSNR, SSIM, LPPIPS = cal_vis_metric(unpruned_videos_path, videos_path)
    my_VBench.evaluate(videos_path=videos_path, name=name)
    # ========== 读取 all_eval_results.json ==========
    eval_json_path = osp.join(result_dir, f"{name}_eval_results.json")

    with open(eval_json_path, "r") as f:
        eval_data = json.load(f)

    # upload_data: dimension → raw mean score
    upload_data = {}

    for key, value in eval_data.items():
        if isinstance(value, list):
            upload_data[key.replace("_", " ")] = value[0]
        else:
            upload_data[key.replace("_", " ")] = value

    # 确保所有维度都有
    for key in TASK_INFO:
        if key not in upload_data:
            upload_data[key] = 0.0

    # ======== 官方 total-score 计算 ========
    normalized_score = get_normalized_score(upload_data)
    quality_score = get_quality_score(normalized_score)
    semantic_score = get_semantic_score(normalized_score)
    final_score = get_final_score(quality_score, semantic_score)

    # ========== 保存最终 summary ==========
    summary_save_path = osp.join(result_dir, "final_summary.json")

    summary = {
        'time':time,
        'memory':memory,
        'PSNR':PSNR,
        'SSIM':SSIM,
        'LPIPS':LPPIPS,
        "quality_score": quality_score,
        "semantic_score": semantic_score,
        "final_score": final_score,
        "raw_scores": upload_data,
        "normalized_scores": normalized_score,

    }

    with open(summary_save_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("====== Final summary saved ======")
    print(summary_save_path)
    print("Final Score:", final_score)


# -----------------------
#   你的主脚本融合 total-score
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="raw",
                    help="实验文件夹名：raw / prune / baseline / ...")
parser.add_argument("--index", type=int, default=0,
                    help="vbench index")
parser.add_argument("--mode", type=str, default="raw",
                    help="mode")
parser.add_argument("--start_mode", type=int, default=0,
                    help="vbench start index")
arg = parser.parse_args()





videos_path = f"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/dongchengqi/InfinityStar/TTM-dev/evaluation/{arg.exp_name}"
mode = os.path.basename(videos_path)
name = "all"

result_dir = f"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/dongchengqi/InfinityStar/TTM-dev/evaluation/results/{arg.exp_name}"
os.makedirs(result_dir, exist_ok=True)
#["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "object_class", "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style", 'overall_consistency', "human_action", "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style"]        
my_VBench = VBench(
    'cuda',
    "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/dongchengqi/InfinityStar/TTM-dev/evaluation/VBench_rewrited_prompt.json",
    result_dir
   
)

# ----------- 执行评估 -----------
my_VBench.evaluate(videos_path=videos_path, name=name
                #    , dimension_list = ["motion_smoothness"]
                   )

# ========== 读取 all_eval_results.json ==========
eval_json_path = osp.join(result_dir, f"{name}_eval_results.json")

with open(eval_json_path, "r") as f:
    eval_data = json.load(f)

# upload_data: dimension → raw mean score
upload_data = {}

for key, value in eval_data.items():
    if isinstance(value, list):
        upload_data[key.replace("_", " ")] = value[0]
    else:
        upload_data[key.replace("_", " ")] = value

# 确保所有维度都有
for key in TASK_INFO:
    if key not in upload_data:
        upload_data[key] = 0.0

# ======== 官方 total-score 计算 ========
normalized_score = get_normalized_score(upload_data)
quality_score = get_quality_score(normalized_score)
semantic_score = get_semantic_score(normalized_score)
final_score = get_final_score(quality_score, semantic_score)

# ========== 保存最终 summary ==========
summary_save_path = osp.join(result_dir, "final_summary.json")

summary = {
    "raw_scores": upload_data,
    "normalized_scores": normalized_score,
    "quality_score": quality_score,
    "semantic_score": semantic_score,
    "final_score": final_score,
}

with open(summary_save_path, "w") as f:
    json.dump(summary, f, indent=4)

print("====== Final summary saved ======")
print(summary_save_path)
print("Final Score:", final_score)
