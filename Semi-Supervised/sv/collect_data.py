import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.LogUtil import setup_logger
from common.MatchUtil import task_sample_match
from typing import List, Literal
logger = setup_logger(__name__)

def collect_OncoTop_sv(
    input_file: str = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML_nochip/OncoTop_SV.tsv",
    output_dir: str = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML_nochip",
    needed_columns: List[str] = ["sampleID", "FusionGene","FusionExon", "BamPath", "Pos1", "Pos2","Freq","FusionType"]
):
    df = pd.DataFrame()
    if input_file.endswith(".xlsx") or input_file.endswith(".xls"):
        df_dict = pd.read_excel(input_file, sheet_name=None)
        outdir = output_dir
        for sheet_name, df in df_dict.items():
            if sheet_name == "SV":
                df = df.loc[:, needed_columns]
                df = df.dropna(subset=needed_columns, how="any")
                outfile = os.path.join(outdir, f"{sheet_name}.tsv")
                df.to_csv(outfile, sep="\t", index=False)
    else:
        df = pd.read_csv(input_file, sep="\t")
        df = df.loc[:, needed_columns]
        df = df.dropna(subset=needed_columns, how="any")
        outfile = os.path.join(output_dir, "SV_processed.tsv")
        df.to_csv(outfile, sep="\t", index=False)
    logger.info(f"Processed OncoTop SV data written to: {output_dir}")
    return df

def collect_ddPCR_data(
    ddPCR_file:str,
    map_file:str,
    outprefix:str,
    ref_file:str = "/mnt/GenePlus002/prod/path2list/all.list",
    ddPCR_origin_col:str = "原始编号",
    map_originID_col:str = "原始编号",
    map_sequenceID_col:str = "核酸编号",
    map_taskID_col:str = "分析单号/路径"
):
    df_ddPCR = pd.read_csv(ddPCR_file, sep="\t")
    df_map = pd.read_csv(map_file, sep="\t")
    df_map = df_map.loc[df_map[map_originID_col].isin(df_ddPCR[ddPCR_origin_col]),[map_sequenceID_col, map_taskID_col]]
    task_samples = []
    sample_dirs = []
    for seq_id, task_id in zip(df_map[map_sequenceID_col], df_map[map_taskID_col]):
        if os.path.exists(task_id):
            logger.info(f"Found task directory")
            sample_dir = os.path.join(task_id,seq_id)
            if os.path.exists(sample_dir):
                sample_dirs.append(sample_dir)
            else:
                logger.warning(f"Sample directory not found for task {task_id}: {sample_dir}")
        else:
            logger.info("task is identifier, not a path, skipping directory check")
            task_samples.append((seq_id, task_id))
    with open(ref_file, "r") as f:
        ref_paths = list(line.strip() for line in f)
    task_list = [task_id for _, task_id in task_samples]
    sample_list = [seq_id for seq_id, _ in task_samples]
    match_results, unmatch_results = task_sample_match(task_list=task_list,sample_list=sample_list,paths=ref_paths)
    sample_dirs.extend(match_results)
    with open(f"{outprefix}_unmatch_results.tsv", "w") as f:
        f.write("sample_id\ttask_id\n")
        for seq_id, task_id in unmatch_results:
            f.write(f"{seq_id}\t{task_id}\n")
    logger.info(f"Matched {len(match_results)} samples, unmatched {len(unmatch_results)} samples")
    with open(f"{outprefix}_sample_dirs.txt", "w") as f:
        for sample_dir in sample_dirs:
            f.write(sample_dir + "\n")

def judge_ddPCR_data(
    sv_file:str,
    ddPCR_file:str,
    map_file:str,
    outprefix:str,
    map_originID_col:str = "原始编号",
    map_sequenceID_col:str = "核酸编号",
    sv_sampleID_col:str = "sampleID",
    merge_keys: List[str] = ["原始编号", "FusionGene", "FusionExon"]
):
    df_map = pd.read_csv(map_file, sep="\t")
    seq2origin = dict(zip(df_map[map_sequenceID_col], df_map[map_originID_col]))
    df_sv = pd.read_csv(sv_file, sep="\t")
    df_sv["原始编号"] = df_sv[sv_sampleID_col].map(seq2origin)
    df_ddPCR = pd.read_csv(ddPCR_file, sep="\t")
    df_sv_joined = df_sv.merge(
        df_ddPCR[merge_keys], on=merge_keys, how="left", indicator=True
    )
    # SV records with matching ddPCR (also includes ddPCR columns from the merge)
    df_matched = df_sv_joined.loc[df_sv_joined["_merge"] == "both"].drop(columns=["_merge"])
    # SV records without matching ddPCR
    df_unmatched = df_sv_joined.loc[df_sv_joined["_merge"] == "left_only"].drop(columns=["_merge"])
    df_res = df_ddPCR.merge(df_sv, left_on=merge_keys, right_on=merge_keys, how="left")
    df_res.to_csv(f"{outprefix}_ddPCR.tsv", sep="\t", index=False)
    df_unmatched.to_csv(f"{outprefix}_no_ddPCR.tsv", sep="\t", index=False)
    logger.info(f"df_sv total: {len(df_sv)}, matched: {len(df_matched)}, unmatched: {len(df_unmatched)}, sum check: {len(df_matched) + len(df_unmatched)}")
        

if __name__ == "__main__":
    # collect_OncoTop_sv()
    ddPCR_file="/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/data/sv/ddPCR.tsv"
    map_file="/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/data/sv/origin_sequenceID.tsv"
    # collect_ddPCR_data(
    #     ddPCR_file=ddPCR_file,
    #     map_file=map_file,
    #     outprefix="/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML_nochip/ddPCR"
    # )
    sv_file = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML_nochip/SV_processed.tsv"
    outprefix = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML_nochip/SV_processed"
    judge_ddPCR_data(
        sv_file=sv_file,
        ddPCR_file=ddPCR_file,
        map_file=map_file,
        outprefix=outprefix
    )
