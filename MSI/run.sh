#!/bin/bash
#? -D beijing/centos7/cnc/apps/onco_fusion/ncarriba@sha256:0f39f91f899eb90158b2f2327bcff1dc192d0a862810b778fa6bfe1383ff2dca
#? -R cpu=10,num_proc=10,mem=50G,max_retries=1,gpu=0,timeout=240
#? -L /mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/jsub/logs/MSI
#? -F /mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/jsub/stage_flags/MSI
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# python ${SCRIPT_DIR}/MSI.py train \
#        --all-info /mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/all_info_dedup.tsv \
#        --output-dir /mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/results/entropy \
#        --locus-selector auc \
#        --selector twostage \
#        --site-file-col site_feature \
#        --auc-threshold 0.80 \
#        --detector xgboost \
#        --threshold-method youden \
#        --use-renqun-mss \
#         --cache-dir /mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/feature
# python ${SCRIPT_DIR}/msi_pct_train.py \
#     --input /mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/msisensor_pro_merged.tsv \
#     -o /mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/results/MSIsensor-pro/mct \
#     --threshold-method youden \
#     --msi-col MSI_real \
#     --use-renqun-mss


# python ${SCRIPT_DIR}/compare_features.py \
#     --all-info /mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/all_info.tsv \
#     --output-dir /mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/results/compare/compare_features \
#     --cache-dir /mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/feature

IHC=/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/wangke_15226556036/project/TopMSIv2.2/IHC_path.xlsx
renqun=/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/wangke_15226556036/project/TopMSIv2.2/人群及独立验证样本.xlsx
# python ${SCRIPT_DIR}/colletc_data.py \
#     -i BL:/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/wangke_15226556036/project/TopMSIv2.2/LR_per/BL_predictions.xlsx \
#     -i renqun:/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/wangke_15226556036/project/TopMSIv2.2/LR_per/renqun_predictions.xlsx \
#     -o /mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/all_info_new.tsv



