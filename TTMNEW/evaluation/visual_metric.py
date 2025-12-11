import os
import cv2
import argparse
import torch
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure
import numpy as np
from tqdm import tqdm
cpu_num = 2
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.multiprocessing.set_sharing_strategy('file_system')
def load_video_frames(path, resize_to=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if resize_to is not None:
            img = cv2.resize(img, resize_to)
        frames.append(np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0))
    cap.release()
    return np.concatenate(frames)


def compute_video_metrics(frames_gt, frames_gen,
                          device, ssim_metric, lpips_fn):
    gt_t = torch.from_numpy(frames_gt).float().to(device).permute(0, 3, 1, 2).div_(255)
    gen_t = torch.from_numpy(frames_gen).float().to(device).permute(0, 3, 1, 2).div_(255)

    mse = torch.mean((gt_t - gen_t) ** 2)
    psnr = -10.0 * torch.log10(mse)

    ssim_val = ssim_metric(gen_t, gt_t)

    with torch.no_grad():
        lpips_val = lpips_fn(gt_t * 2.0 - 1.0, gen_t * 2.0 - 1.0).mean()

    return psnr.item(), ssim_val.item(), lpips_val.item()

def cal_vis_metric(gt_folder, gen_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 🔍 找出两个文件夹中同名的视频
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(".mp4")])
    gen_files = sorted([f for f in os.listdir(gen_folder) if f.endswith(".mp4")])

    common_files = sorted(set(gt_files) & set(gen_files))

    if not common_files:
        print("两个文件夹没有同名视频")
        return

    psnrs, ssims, lpips_vals = [], [], []
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net="alex", spatial=True).to(device)
    for fname in tqdm(common_files, desc="Evaluating Videos"):
        path_gt = os.path.join(gt_folder, fname)
        path_gen = os.path.join(gen_folder, fname)

        frames_gt = load_video_frames(path_gt)
        frames_gen = load_video_frames(path_gen)

        # 自动 resize 生成的视频使分辨率一致
        if frames_gt.shape[2:] != frames_gen.shape[2:]:
            h, w = frames_gt.shape[2], frames_gt.shape[3]
            frames_gen = load_video_frames(path_gen, resize_to=(w, h))

        p, s, l = compute_video_metrics(frames_gt, frames_gen,
                                        device, ssim_metric, lpips_fn)

        psnrs.append(p)
        ssims.append(s)
        lpips_vals.append(l)

    print("\n=== Overall Averages ===")
    print(f"Average PSNR : {np.mean(psnrs):.2f} dB")
    print(f"Average SSIM : {np.mean(ssims):.4f}")
    print(f"Average LPIPS: {np.mean(lpips_vals):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_video",  default="/data3/chengqidong/mrg/InfinityStar/visual/new_raw", help="GT videos folder")
    parser.add_argument("--generated_video",  default="/data3/chengqidong/mrg/InfinityStar/visual/skip2_28", help="Generated videos folder")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    

    gt_folder = args.original_video
    gen_folder = args.generated_video
    cal_vis_metric(gt_folder, gen_folder)
    


if __name__ == "__main__":
    main()
