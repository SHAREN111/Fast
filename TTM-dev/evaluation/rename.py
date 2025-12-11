import os
import os.path as osp
import json
from typing import List
import re

def load_refined_prompts(json_path: str):
    """返回 [(refined_prompt, raw_prompt)] 列表"""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
        return [(item['refined_prompt'], item['prompt_en']) for item in data]

def sanitize_filename(name: str) -> str:
    """把 prompt 变成安全文件名"""
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]', '', name)  # 删除非法字符
    name = re.sub(r'\s+', '_', name)
    return name[:180]

# ======================================
# 你的目录和 JSON 路径
# ======================================
save_dir = '/data3/chengqidong/mrg/InfinityStar/evaluation'
raw_dir = osp.join(save_dir, 'TTM_10')
os.makedirs(raw_dir, exist_ok=True)

prompts = load_refined_prompts(
    '/data3/chengqidong/mrg/InfinityStar/evaluation/VBench_rewrited_prompt.json'
)

# ======================================
# 只筛选纯数字文件，例如 0.mp4，12.mp4
# ======================================
video_files = [
    f for f in os.listdir(raw_dir)
    if f.endswith(".mp4") and re.fullmatch(r'\d+\.mp4', f)
]

print(f"检测到 {len(video_files)} 个需要重命名的旧文件（纯数字形式）")

# 记录 idx → 文件名
idx_to_old = {}

for fname in video_files:
    idx = int(fname.split('.')[0])
    idx_to_old[idx] = fname

# ======================================
# 开始重命名
# ======================================
for idx, dual_prompt in enumerate(prompts):
    # 只重命名存在 idx 的文件
    if idx not in idx_to_old:
        continue

    refined_prompt, raw_prompt = dual_prompt
    old_name = idx_to_old[idx]
    old_path = osp.join(raw_dir, old_name)

    #clean_prompt = sanitize_filename(raw_prompt)
    new_name = f"{raw_prompt}-0.mp4"
    new_path = osp.join(raw_dir, new_name)

    # 新文件已存在 => 避免覆盖，跳过
    if osp.exists(new_path):
        print(f"[SKIP] {new_name} 已存在（可能已生成）")
        continue

    # 执行重命名
    os.rename(old_path, new_path)
    print(f"[RENAME] {old_name} → {new_name}")

print("重命名完成！")
