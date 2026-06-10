import pandas as pd
from typing import List
def combine_data(
    infile1:str,
    infile2:str,
    outfile:str,
    target_cols:List[str]= ["原始编号","BamPath", "Pos1", "Pos2", "mutation", "Freq", "ddPCR_AF"],
):
    df1 = pd.read_csv(infile1, sep="\t")
    df2 = pd.read_csv(infile2, sep="\t")
    if not all(col in df1.columns for col in target_cols):
        raise ValueError(f"infile1 must contain columns: {target_cols}")
    if not all(col in df2.columns for col in target_cols):
        raise ValueError(f"infile2 must contain columns: {target_cols}")
    df_combined = pd.concat([df1[target_cols], df2[target_cols]], ignore_index=True)
    df_combined.rename(columns={"原始编号": "Original_ID"}, inplace=True)
    df_combined.to_csv(outfile, sep="\t", index=False)
    return df_combined

if __name__ == "__main__":
    infile1 = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/data/Sheet1.tsv"
    infile2 = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/data/SV_jiaozheng_yanzheng.tsv"
    outfile = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/data/combined_data.tsv"
    df_combined = combine_data(infile1, infile2, outfile)
    