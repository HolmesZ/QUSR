# 计算 FID 指标

import argparse
import os
from pytorch_fid import fid_score


def calc_fid(paths, batch_size=1, device="cuda", dims=2048):
    return fid_score.calculate_fid_given_paths(paths, batch_size, device, dims)


# rslt_path = "/data/mfe/FastDiffSR/FID/FastDiffSR_potsdam_x4"
# rslt_path = "/data/mfe/FastDiffSR/FID/FastDiffSR_potsdam_x8"
# hr_path = "/data/mfe/FastDiffSR/MSI_SR_model_4/dataset/Test/Potsdam"
# rslt_path = "/data/mfe/FastDiffSR/FID/FastDiffSR_toronto_x4"
# rslt_path = "/data/mfe/FastDiffSR/FID/FastDiffSR_toronto_x8"
# hr_path = "/data/mfe/FastDiffSR/MSI_SR_model_4/dataset/Test/Toronto"

def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID between SR results and HR ground truth.")
    parser.add_argument("--sr", required=True, help="SR results directory.")
    parser.add_argument("--hr", required=True, help="HR ground-truth directory.")
    parser.add_argument("--suffix", default="_sr.tif", help="SR filename suffix to include.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for FID.")
    parser.add_argument("--device", default="cuda", help="Device for FID.")
    parser.add_argument("--dims", type=int, default=2048, help="Inception dims.")
    parser.add_argument("--link-dir", default=None, help="Optional dir to store symlinks.")
    return parser.parse_args()


def prepare_sr_dir(sr_path, suffix, link_dir=None):
    use_dir = link_dir or f"{sr_path}_fid"
    os.makedirs(use_dir, exist_ok=True)
    for filename in os.listdir(sr_path):
        if filename.endswith(suffix):
            src_file = os.path.join(sr_path, filename)
            dst_file = os.path.join(use_dir, filename)
            if not os.path.exists(dst_file):
                os.symlink(src_file, dst_file)  # 使用符号链接节省空间
    return use_dir


def main():
    args = parse_args()
    sr_dir = prepare_sr_dir(args.sr, args.suffix, args.link_dir)
    paths = [sr_dir, args.hr]
    fid_score_result = calc_fid(paths, args.batch, args.device, args.dims)
    print("- SR_FID : {:.3f}".format(fid_score_result))


if __name__ == "__main__":
    main()
