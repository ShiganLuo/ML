# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.LogUtil import setup_logger
from common.CmdUtil import _run_cmd
import pandas as pd
from typing import Dict, Optional
import subprocess
logger = setup_logger(__name__)

def run_MSIsensor_Pro_wrapper(
    infile:str,
    outdir:str,
    out_scan:str,
    msisensor_pro:str,
    fasta:Optional[str] = None,
    bam_path_col:str = "bam_path",
    dry_run:bool = False
):
    df = pd.read_csv(infile, sep="\t")
    out_res_dir = os.path.join(outdir, "msisensor_pro_results")
    os.makedirs(out_res_dir, exist_ok=True)
    out_script_dir = os.path.join(outdir, "msisensor_pro_scripts")
    os.makedirs(out_script_dir, exist_ok=True)
    df = df.drop_duplicates(subset=[bam_path_col])
    df = df.dropna(subset=[bam_path_col])
    logger.info(f"Total samples to process: {len(df)}")
    for idx, row in df.iterrows():
        bam_tumor = row[bam_path_col]
        sample_id = os.path.basename(bam_tumor).split("_cancer")[0]
        if not os.path.exists(bam_tumor):
            logger.warning(f"BAM file not found for sample {sample_id}: {bam_tumor}. Skipping this sample.")
            continue
        if not os.path.exists(out_scan):
            logger.warning(f"MSIsensor-Pro scan result file not found: {out_scan}, will run scan.")
            if not os.path.exists(fasta):
                logger.error(f"Reference FASTA file not found: {fasta}. Cannot run MSIsensor-Pro scan.")
                return
            cmd_scan = [msisensor_pro, "scan", "-d", fasta , "-o", out_scan]
            _run_cmd(cmd_scan, logger)
        sample_out_dir = os.path.join(out_res_dir, sample_id)
        os.makedirs(sample_out_dir, exist_ok=True)
        out_msi = os.path.join(sample_out_dir, f"{sample_id}.msi")
        cmd_msi = [msisensor_pro, "pro", "-d", out_scan, "-t", bam_tumor, "-o", out_msi]
        out_script_path = os.path.join(out_script_dir, f"{sample_id}_msisensor_pro.sh")
        logs = os.path.join(outdir, f"status/logs/{sample_id}_msisensor_pro.log")
        os.makedirs(os.path.dirname(logs), exist_ok=True)
        flags = os.path.join(outdir, f"status/flags/{sample_id}_msisensor_pro.flag")
        os.makedirs(os.path.dirname(flags), exist_ok=True)
        with open(out_script_path, "w", encoding="utf-8") as fh:
            fh.write("#!/bin/bash\n")
            fh.write("#? -D beijing/centos7/cnc/base@sha256:0155936d325fb0c985752683b67de65dea8644676c5b5a0e1727faa9f39858f9\n")
            fh.write("#? -R cpu=1,num_proc=1,mem=1G,max_retries=1,gpu=0,timeout=2400\n")
            fh.write(f"#? -L {logs}\n")
            fh.write(f"#? -F {flags}\n")
            fh.write("set -euo pipefail\n\n")
            fh.write(" ".join(cmd_msi) + "\n")
        logger.info(f"MSIsensor-Pro command for sample {sample_id} written to: {out_script_path}")
        if not dry_run:
            run_cmd = ["jsub.py", out_script_path]
            logger.info(f"submit  MSIsensor-Pro  task for sample {sample_id} with command: {' '.join(run_cmd)}")
            _run_cmd(run_cmd, logger)
        else:
            logger.info(f"Dry run enabled, skipping execution of MSIsensor-Pro for sample {sample_id}")

if __name__ == "__main__":
    infile = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/all_info.tsv"
    outdir = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/MSIsensor-pro"
    fasta = "/mnt/DB-linjian/DB/pubDB/human/hg19/refs/tgp_phase2_flat/phase2_flat/hs37d5.fa"
    out_scan = os.path.join(outdir, "msisensor_pro_scan_result.hs37d5.bed")
    run_MSIsensor_Pro_wrapper(
        infile=infile,
        outdir=outdir,
        fasta=fasta,
        out_scan=out_scan,
        msisensor_pro="/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/miniforge3/envs/MSI/bin/msisensor-pro",
        bam_path_col="bam_path",
        dry_run=False
    )



