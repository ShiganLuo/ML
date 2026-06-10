#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

labeled_tsv=/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML_nochip/SV_processed_ddPCR.tsv
unlabeled_tsv=/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML_nochip/SV_processed_no_ddPCR.tsv
outdir=/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML_nochip

python ${SCRIPT_DIR}/sv_freq_correction/consistency_reg.py \
    --labeled-tsv ${labeled_tsv} \
    --unlabeled-tsv ${unlabeled_tsv} \
    -o ${outdir}/consistency_reg \
    --consistency-weight 1.0 \
    --noise-std 0.1 \
    --feature-dir ${outdir}/features