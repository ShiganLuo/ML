# Author: ShiganLuo
# Email: 25303020102@qq.com
# GitHub: https://github.com/ShiganLuo
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.LogUtil import setup_logger
import glob
import argparse
from typing import List, Dict, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
logger = setup_logger("collect_data")

# ---------------------------------------------------------------------------
# BAM path resolution functions
# ---------------------------------------------------------------------------
def _parallel_apply(series, func, max_workers=16):
    """Apply func to each element of series in parallel using threads.

    Parameters
    ----------
    series : pd.Series
        Input series to apply func on.
    func : callable
        Function to apply to each element.
    max_workers : int, optional
        Number of threads. Default: 16.

    Returns
    -------
    pd.Series
        Result series with the same index.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(func, v): i for i, v in series.items()}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return pd.Series(results, index=series.index)


def resolve_bam_bl(site_path: str, bl_site_to_bam_map: Optional[dict] = None,
                   bl_suffix_map: Optional[dict] = None) -> Optional[str]:
    """Resolve BAM path from a BL-type site_path.

    Applies substring replacement (site fragment -> bam fragment) and
    suffix replacement (.site.txt -> .bam) to derive the BAM file path.
    Returns None if the resulting path does not exist on disk.

    Parameters
    ----------
    site_path : str
        Path to the site feature file (e.g. a .site.txt file).
    bl_site_to_bam_map : dict, optional
        Mapping of substrings to replace in the path.
        Default: ``{"feature/TopMSI_BaselineSample/cancer": "baseline_bam"}``.
    bl_suffix_map : dict, optional
        Mapping of file suffixes to replace.
        Default: ``{".site.txt": ".bam"}``.

    Returns
    -------
    str or None
        Resolved BAM file path if it exists, otherwise None.

    Examples
    --------
    >>> site_path = ".../TopMSIv2.2/feature/TopMSI_BaselineSample/cancer/x.site.txt"
    >>> resolve_bam_bl(site_path)
    ".../TopMSIv2.2/baseline_bam/cancer/x.bam"
    """
    if bl_site_to_bam_map is None:
        bl_site_to_bam_map = {"feature/TopMSI_BaselineSample/cancer": "baseline_bam"}
    if bl_suffix_map is None:
        bl_suffix_map = {".site.txt": ".bam"}

    if not isinstance(site_path, str) or not site_path:
        return None

    bam_path = site_path
    for old, new in bl_site_to_bam_map.items():
        bam_path = bam_path.replace(old, new)
    for old_suffix, new_suffix in bl_suffix_map.items():
        if bam_path.endswith(old_suffix):
            bam_path = bam_path[: -len(old_suffix)] + new_suffix
            break

    if os.path.isfile(bam_path) and os.path.getsize(bam_path) > 0:
        return bam_path
    return None


def resolve_bam_common(
        CNC_path: str, 
        bam_subdir: str = "cancer/4_realign_bam",
        bam_pattern: str = "*.bam") -> Optional[str]:
    """Resolve BAM path from a PCR or renqun-type CNC path.

    Constructs the path ``{CNC_path}/{bam_subdir}/{bam_pattern}``
    and returns the first matching BAM file (sorted by filename).
    Returns None if no BAM file is found.

    Parameters
    ----------
    CNC_path : str
        Root CNC directory path.
    bam_subdir : str, optional
        Subdirectory under the CNC path containing BAM files.
        Default: ``"cancer/4_realign_bam"``.
    bam_pattern : str, optional
        Glob pattern for BAM files. Default: ``"*.bam"``.

    Returns
    -------
    str or None
        Path to the first matching BAM file, or None if not found.
    """
    if not isinstance(CNC_path, str) or not CNC_path:
        return None
    bam_path = None
    bam_dir = os.path.join(CNC_path, bam_subdir)
    bam_pattern = os.path.join(bam_dir, bam_pattern)
    candidates = glob.glob(bam_pattern)

    if candidates:
        candidates.sort()
        for c in candidates:
            if os.path.exists(c) and os.path.getsize(c) > 0:
                bam_path = c
                break

    return bam_path

def resolve_bam_dir(
    sample_id: str,
    backup_dir: str = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/wangke_15226556036/project/TopMSIv2.2/baseline_bam",
    bam_pattern: str = "*.bam",
):
    """
    Resolve BAM path from a backup directory based on sample ID.
    """
    if not isinstance(sample_id, str) or not sample_id:
        return None
    backup_path = os.path.join(backup_dir, sample_id)
    bam_files = glob.glob(os.path.join(backup_path, bam_pattern))
    if bam_files:
        bam_files.sort()
        for bam_file in bam_files:
            if os.path.exists(bam_file) and os.path.getsize(bam_file) > 0:
                return bam_file
    return None

def resolve_bam_for_bl(site_path: str,
                       bl_prefix: str = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/wangke_15226556036/project/TopMSIv2.2",
                       bl_site_to_bam_map: Optional[dict] = None,
                       bl_suffix_map: Optional[dict] = None,
                       bam_subdir: str = "cancer/4_realign_bam",
                       bam_pattern: str = "*.bam") -> Optional[str]:
    """Dispatch BAM resolution for BL-type data.

    If ``site_path`` contains ``bl_prefix``, delegates to
    :func:`resolve_bam_bl`; otherwise falls back to
    :func:`resolve_bam_common`.

    Parameters
    ----------
    site_path : str
        Path to the site feature file.
    bl_prefix : str, optional
        Prefix string used to identify BL-origin paths.
    bl_site_to_bam_map : dict, optional
        Substring replacement mapping passed to :func:`resolve_bam_bl`.
        Default: ``{"feature/TopMSI_BaselineSample/cancer": "baseline_bam"}``.
    bl_suffix_map : dict, optional
        Suffix replacement mapping passed to :func:`resolve_bam_bl`.
        Default: ``{".site.txt": ".bam"}``.
    bam_subdir : str, optional
        BAM subdirectory passed to :func:`resolve_bam_common`.
        Default: ``"cancer/4_realign_bam"``.
    bam_pattern : str, optional
        Glob pattern passed to :func:`resolve_bam_common`.
        Default: ``"*.bam"``.

    Returns
    -------
    str or None
        Resolved BAM file path, or None if not found.
    """
    if not isinstance(site_path, str) or not site_path:
        return None
    if bl_prefix in site_path:
        return resolve_bam_bl(site_path, bl_site_to_bam_map, bl_suffix_map)
    return resolve_bam_common(site_path, bam_subdir, bam_pattern)


def add_bam_column(df: pd.DataFrame, origin: str, args: argparse.Namespace) -> pd.DataFrame:
    """Add a ``bam_path`` column to the DataFrame based on origin type.

    Dispatches to the appropriate BAM resolver for each row using the
    CRC column (for PCR/renqun) or the site_path column (for BL).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the source columns.
    origin : str
        Dataset origin type: ``"PCR"``, ``"renqun"``, or ``"BL"``.
    args : argparse.Namespace
        Parsed command-line arguments containing column names and
        BAM resolution configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame with the added ``bam_path`` column.

    Raises
    ------
    ValueError
        If ``origin`` is not one of ``"PCR"``, ``"renqun"``, or ``"BL"``.
    """
    if origin in ("PCR", "renqun"):
        df["bam_path"] = df[args.col_crc].apply(
            lambda x: resolve_bam_common(x, args.bam_subdir, args.bam_pattern))
    elif origin == "BL":
        df["bam_path"] = df[args.col_site_path].apply(
            lambda x: resolve_bam_for_bl(
                x, args.bl_prefix, args.bl_site_to_bam_map,
                args.bl_suffix_map, args.bam_subdir, args.bam_pattern))
    else:
        raise ValueError(f"Unknown origin: {origin}")

    return df


def combine_datasets(
    in_dict: Dict[str, str],
    outfile: str,
    args: argparse.Namespace,
    keep_columns: Optional[List[str]] = None,
):
    """Combine multiple datasets into a single TSV with BAM path annotation.

    Reads each input Excel file, adds ``origin`` and ``bam_path`` columns,
    renames MSI-related columns to a unified schema, and writes the merged
    result to ``outfile``.

    Parameters
    ----------
    in_dict : dict
        Mapping of origin type to input file path
        (e.g. ``{"BL": "/path/to/bl.xlsx", "PCR": "/path/to/pcr.xlsx"}``).
    outfile : str
        Output TSV file path.
    args : argparse.Namespace
        Parsed command-line arguments with BAM resolution config.
    keep_columns : list of str, optional
        Columns to retain in the output. Default:
        ``["origin", "bam_path", "site_path", "site_feature",
        "MSI_CNC", "MSI_real", "cancertype"]``.

    Returns
    -------
    None
    """
    if keep_columns is None:
        keep_columns = [
            "origin", "bam_path", "site_path", "site_feature",
            "MSI_CNC", "MSI_real", "cancertype",
        ]

    logger = setup_logger("combine_datasets")
    df_list = []
    for origin, infile in in_dict.items():
        df_dict = pd.read_excel(infile, sheet_name=None)
        for sheet_name, df in df_dict.items():
            df["origin"] = origin
            df = add_bam_column(df, origin, args)
            if origin == "BL":
                df.rename(columns={"MSI_flag": "MSI_real", "MSI_status": "MSI_CNC"}, inplace=True)
            elif origin == "PCR":
                df.rename(columns={"PCR-MSI状态": "MSI_real", "MSI_status": "MSI_CNC"}, inplace=True)
            else:
                df.rename(columns={"MSI_status": "MSI_CNC"}, inplace=True)
            for col in keep_columns:
                if col not in df.columns:
                    df[col] = None
            df = df[keep_columns]
            df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(outfile, sep="\t", index=False)

    total = len(combined_df)
    found = combined_df["bam_path"].notna().sum()
    missing = total - found
    logger.info(f"Total={total}, BAM found={found}, BAM missing={missing}")
    logger.info(f"Output: {outfile}")


def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with input files, output path, and BAM
        resolution configuration.
    """
    parser = argparse.ArgumentParser(
        description="合并多个数据集，补充 origin 和 bam_path 列后输出。")

    # 输入/输出
    parser.add_argument(
        "-i", "--input", dest="inputs", action="append", required=True,
        help="输入文件，格式为 origin:path (如 BL:/path/to/file.xlsx)，可多次指定")
    parser.add_argument(
        "-o", "--output", required=True,
        help="输出 TSV 文件路径")

    # BL 配置
    parser.add_argument(
        "--bl-prefix",
        default="/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/wangke_15226556036/project/TopMSIv2.2",
        help="BL 类型前缀路径，匹配时用 resolve_bam_bl，否则回退到 resolve_bam_common")
    parser.add_argument(
        "--bl-site-to-bam-map", action="append",
        default=["feature/TopMSI_BaselineSample/cancer:baseline_bam"],
        help="BL site→bam 映射，格式 old:new，可多次指定")
    parser.add_argument(
        "--bl-suffix-map", action="append",
        default=[".site.txt:.bam"],
        help="BL 后缀映射，格式 old_suffix:new_suffix，可多次指定")

    # PCR / renqun 配置
    parser.add_argument(
        "--bam-subdir", default="cancer/4_realign_bam",
        help="PCR bam 子目录 (默认: cancer/4_realign_bam)")
    parser.add_argument(
        "--bam-pattern", default="*.bam",
        help="PCR bam glob 模式 (默认: *.bam)")

    # 列名
    parser.add_argument(
        "--col-site-path", default="site_path",
        help="site_path 列名 (默认: site_path)")
    parser.add_argument(
        "--col-crc", default="CNC",
        help="CRC 列名 (默认: CNC)")

    return parser.parse_args()


def main():
    """Entry point: parse args, merge datasets, and report statistics."""
    args = parse_args()

    # 解析 action="append" 的映射参数为 dict
    bl_site_to_bam_map = {}
    for item in args.bl_site_to_bam_map:
        old, new = item.split(":", 1)
        bl_site_to_bam_map[old] = new
    args.bl_site_to_bam_map = bl_site_to_bam_map

    bl_suffix_map = {}
    for item in args.bl_suffix_map:
        old, new = item.split(":", 1)
        bl_suffix_map[old] = new
    args.bl_suffix_map = bl_suffix_map

    # 解析 -i origin:path
    in_dict = {}
    for item in args.inputs:
        origin, path = item.split(":", 1)
        in_dict[origin] = path

    combine_datasets(in_dict, args.output, args)

    logger = setup_logger("main")
    df = pd.read_csv(args.output, sep="\t")
    logger.info(f"Total rows: {len(df)}")
    df["sample_id"] = df["site_feature"].apply(
        lambda x: os.path.basename(x).split("_cancer")[0] if isinstance(x, str) else None)
    dup = df[df["sample_id"].duplicated(keep="first")]
    if len(dup):
        logger.info(f"Duplicated samples:\n{dup}")
    logger.info(f"After dedup: {df.drop_duplicates(subset=['sample_id']).shape}")

def run():
    logger = setup_logger("run")
    in_dict = {
        "BL":  "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/data/MSI/IHC_path.xlsx",
        "renqun":  "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/data/MSI/人群及独立验证样本.xlsx"
    }
    df_dict = {}
    for origin, in_excel in in_dict.items():
        if not os.path.isfile(in_excel):
            raise FileNotFoundError(f"Input file for {origin} not found: {in_excel}")
        if origin == "BL":
            df = pd.read_excel(in_excel, sheet_name="Sheet1")
            df_dict[origin] = df
        elif origin == "renqun":
            df = pd.read_excel(in_excel, sheet_name="renqun")
            df_dict[origin] = df
        else:
            raise ValueError(f"Unknown origin: {origin}")
    outfile = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260615_MSI/output/MSI/data/all_info_v2.tsv"
    keep_columns = ["样本编号","解读匹配癌种", "肿瘤细胞含量","TMB状态","MSI状态","CNC", "site_path"]
    in_list = []
    for origin, df in df_dict.items():
        if origin == "BL":
            cols = keep_columns + ["IHC"]
            df = df[cols]
            df["origin"] = origin
            df["bam_path"] = _parallel_apply(df["CNC"], resolve_bam_common)
            mask = df["bam_path"].isna()
            df.loc[mask, "bam_path"] = _parallel_apply(
                df.loc[mask, "样本编号"], resolve_bam_dir)
        elif origin == "renqun":
            df = df[keep_columns]
            df["origin"] = origin
            df["bam_path"] = _parallel_apply(df["CNC"], resolve_bam_common)
            mask = df["bam_path"].isna()
            df.loc[mask, "bam_path"] = _parallel_apply(
                df.loc[mask, "样本编号"], resolve_bam_dir)
        in_list.append(df)
    df_combined = pd.concat(in_list, ignore_index=True)

    # Derive site_feature from site_path + origin
    _FEATURE_BASE = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/wangke_15226556036/project/TopMSIv2.2/feature"
    _ORIGIN_SUBDIR = {"BL": "BL_addfreq", "renqun": "Frequency", "PCR": "PCR"}
    def _derive_site_feature(row):
        sp = row.get("site_path")
        orig = row.get("origin")
        if not isinstance(sp, str) or not sp or not isinstance(orig, str):
            return None
        subdir = _ORIGIN_SUBDIR.get(orig)
        if not subdir:
            return None
        return os.path.join(_FEATURE_BASE, subdir, os.path.basename(sp))
    df_combined["site_feature"] = df_combined.apply(_derive_site_feature, axis=1)

    # Drop rows where site_path is empty (no site data available)
    before = len(df_combined)
    df_combined = df_combined.dropna(subset=["site_path"])
    df_combined = df_combined[df_combined["site_path"] != ""]
    logger.info(f"Dropped {before - len(df_combined)} rows with empty site_path")

    df_combined.rename(columns={"样本编号": "sample_id", 
                                "解读匹配癌种": "cancertype", 
                                "肿瘤细胞含量": "tumor_content",
                                "TMB状态": "TMB_status", "MSI状态":"MSI_CNC", "CNC": "sample_dir", "site_path": "site_path","IHC": "MSI_IHC"}, inplace=True)
    df_combined = df_combined.drop_duplicates()
    df_combined.to_csv(outfile, sep="\t", index=False)



if __name__ == "__main__":
    # main()
    run()
