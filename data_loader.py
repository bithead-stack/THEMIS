from __future__ import annotations

import os
from dataclasses import dataclass
import ipaddress
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


META_COLS_DAPT = [
    "__row_id",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Activity",
    "Stage",
    "__source_file",
]


def _safe_ip_is_private(s: str) -> int:
    try:
        return int(ipaddress.ip_address(s).is_private)
    except Exception:
        return 0


def _safe_ip_to_int(s: str) -> float:
    try:
        return float(int(ipaddress.ip_address(s)))
    except Exception:
        return np.nan


def _safe_ip_is_multicast(s: str) -> int:
    try:
        return int(ipaddress.ip_address(s).is_multicast)
    except Exception:
        return 0


def _safe_ip_is_loopback(s: str) -> int:
    try:
        return int(ipaddress.ip_address(s).is_loopback)
    except Exception:
        return 0


def _safe_ip_is_link_local(s: str) -> int:
    try:
        return int(ipaddress.ip_address(s).is_link_local)
    except Exception:
        return 0


def _safe_ip_version(s: str) -> int:
    try:
        return int(ipaddress.ip_address(s).version)
    except Exception:
        return 0


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    out = a_num / b_num.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _read_dapt_csv(path: str, canonical_columns: list[str]) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().rstrip("\n")
    if first.startswith("Flow ID,"):
        return pd.read_csv(path, low_memory=False)
    return pd.read_csv(path, header=None, names=canonical_columns, low_memory=False)


def load_dapt2020_dataset(dapt_dir: str) -> pd.DataFrame:
    paths = sorted(
        p for p in (os.path.join(dapt_dir, f) for f in os.listdir(dapt_dir)) if p.endswith(".csv")
    )
    if not paths:
        raise FileNotFoundError(f"No csv found under: {dapt_dir}")

    canonical_columns: list[str] | None = None
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().rstrip("\n")
        if first.startswith("Flow ID,"):
            canonical_columns = first.split(",")
            break
    if canonical_columns is None:
        raise ValueError("Cannot infer canonical header for dapt2020 csvs.")

    frames: list[pd.DataFrame] = []
    for p in paths:
        df = _read_dapt_csv(p, canonical_columns)
        df["__source_file"] = os.path.basename(p)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    merged["Stage"] = merged["Stage"].replace({"BENIGN": "Benign"})
    merged["Activity"] = merged["Activity"].replace({"BENIGN": "Normal"})

    merged = merged.drop_duplicates(subset=["Flow ID", "Timestamp", "Src IP", "Dst IP", "Src Port", "Dst Port"])
    merged = merged.reset_index(drop=True)
    merged["__row_id"] = np.arange(len(merged), dtype=np.int64)
    merged["SrcIsPrivate"] = merged["Src IP"].astype(str).map(_safe_ip_is_private)
    merged["DstIsPrivate"] = merged["Dst IP"].astype(str).map(_safe_ip_is_private)
    merged["SrcIPInt"] = merged["Src IP"].astype(str).map(_safe_ip_to_int)
    merged["DstIPInt"] = merged["Dst IP"].astype(str).map(_safe_ip_to_int)
    merged["SameSubnet24"] = (
        (merged["SrcIsPrivate"] == 1)
        & (merged["DstIsPrivate"] == 1)
        & (merged["SrcIPInt"].fillna(-1).astype(np.int64) // 256 == merged["DstIPInt"].fillna(-2).astype(np.int64) // 256)
    ).astype(np.int64)

    ts = pd.to_datetime(merged["Timestamp"].astype(str), errors="coerce", dayfirst=True)
    merged["TsHour"] = ts.dt.hour.fillna(-1).astype(np.int64)
    merged["TsDayOfWeek"] = ts.dt.dayofweek.fillna(-1).astype(np.int64)
    merged["TsMinuteOfDay"] = (ts.dt.hour.fillna(0) * 60 + ts.dt.minute.fillna(0)).astype(np.int64)
    merged["TsIsWeekend"] = ts.dt.dayofweek.isin([5, 6]).fillna(False).astype(np.int64)

    src_port = pd.to_numeric(merged["Src Port"], errors="coerce")
    dst_port = pd.to_numeric(merged["Dst Port"], errors="coerce")
    merged["SrcPortWellKnown"] = (src_port <= 1023).fillna(False).astype(np.int64)
    merged["DstPortWellKnown"] = (dst_port <= 1023).fillna(False).astype(np.int64)
    merged["DstPortWeb"] = dst_port.isin([80, 443, 8080]).fillna(False).astype(np.int64)
    merged["DstPortSSH"] = dst_port.isin([22]).fillna(False).astype(np.int64)
    merged["DstPortDB"] = dst_port.isin([3306, 5432, 1433]).fillna(False).astype(np.int64)
    merged["DstPort9000"] = dst_port.isin([9000]).fillna(False).astype(np.int64)

    merged["FlowDirInternal"] = (
        (merged["SrcIsPrivate"] == 1) & (merged["DstIsPrivate"] == 1)
    ).astype(np.int64)
    merged["FlowDirOutbound"] = (
        (merged["SrcIsPrivate"] == 1) & (merged["DstIsPrivate"] == 0)
    ).astype(np.int64)
    merged["FlowDirInbound"] = (
        (merged["SrcIsPrivate"] == 0) & (merged["DstIsPrivate"] == 1)
    ).astype(np.int64)
    return merged


def _add_network_context_features(
    df: pd.DataFrame,
    src_ip_col: str,
    dst_ip_col: str,
    src_port_col: str,
    dst_port_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    out = df.copy()

    out["SrcIsPrivate"] = out[src_ip_col].astype(str).map(_safe_ip_is_private)
    out["DstIsPrivate"] = out[dst_ip_col].astype(str).map(_safe_ip_is_private)
    out["SrcIsMulticast"] = out[src_ip_col].astype(str).map(_safe_ip_is_multicast)
    out["DstIsMulticast"] = out[dst_ip_col].astype(str).map(_safe_ip_is_multicast)
    out["SrcIsLoopback"] = out[src_ip_col].astype(str).map(_safe_ip_is_loopback)
    out["DstIsLoopback"] = out[dst_ip_col].astype(str).map(_safe_ip_is_loopback)
    out["SrcIsLinkLocal"] = out[src_ip_col].astype(str).map(_safe_ip_is_link_local)
    out["DstIsLinkLocal"] = out[dst_ip_col].astype(str).map(_safe_ip_is_link_local)
    out["SrcIPVersion"] = out[src_ip_col].astype(str).map(_safe_ip_version)
    out["DstIPVersion"] = out[dst_ip_col].astype(str).map(_safe_ip_version)
    out["SrcIsIPv6"] = (out["SrcIPVersion"] == 6).astype(np.int64)
    out["DstIsIPv6"] = (out["DstIPVersion"] == 6).astype(np.int64)
    out["SameHostIP"] = (out[src_ip_col].astype(str) == out[dst_ip_col].astype(str)).astype(np.int64)
    out["SrcIPInt"] = out[src_ip_col].astype(str).map(_safe_ip_to_int)
    out["DstIPInt"] = out[dst_ip_col].astype(str).map(_safe_ip_to_int)
    out["SameSubnet24"] = (
        (out["SrcIsPrivate"] == 1)
        & (out["DstIsPrivate"] == 1)
        & (out["SrcIPInt"].fillna(-1).astype(np.int64) // 256 == out["DstIPInt"].fillna(-2).astype(np.int64) // 256)
    ).astype(np.int64)

    ts = pd.to_datetime(out[timestamp_col].astype(str), errors="coerce", dayfirst=True)
    out["TsHour"] = ts.dt.hour.fillna(-1).astype(np.int64)
    out["TsDayOfWeek"] = ts.dt.dayofweek.fillna(-1).astype(np.int64)
    out["TsMinuteOfDay"] = (ts.dt.hour.fillna(0) * 60 + ts.dt.minute.fillna(0)).astype(np.int64)
    out["TsIsWeekend"] = ts.dt.dayofweek.isin([5, 6]).fillna(False).astype(np.int64)

    src_port = pd.to_numeric(out[src_port_col], errors="coerce")
    dst_port = pd.to_numeric(out[dst_port_col], errors="coerce")
    src_ip_int = out["SrcIPInt"].fillna(-1).astype(np.int64, copy=False)
    dst_ip_int = out["DstIPInt"].fillna(-1).astype(np.int64, copy=False)
    out["DstPortInt"] = dst_port.fillna(-1).astype(np.int64, copy=False)
    out["SrcPortEphemeral"] = (src_port >= 49152).fillna(False).astype(np.int64)
    out["DstPortEphemeral"] = (dst_port >= 49152).fillna(False).astype(np.int64)
    out["DstPortRegistered"] = ((dst_port >= 1024) & (dst_port <= 49151)).fillna(False).astype(np.int64)
    out["DstPortHigh"] = (dst_port >= 1024).fillna(False).astype(np.int64)

    g_src = out.groupby(src_ip_int, sort=False, dropna=False)
    out["SrcIPFlowCount"] = g_src["__row_id"].transform("count").astype(np.int64, copy=False)
    out["SrcIPUniqueDstPort"] = g_src["DstPortInt"].transform("nunique").astype(np.int64, copy=False)
    out["SrcIPDstPortMin"] = g_src["DstPortInt"].transform("min").astype(np.int64, copy=False)
    out["SrcIPDstPortMax"] = g_src["DstPortInt"].transform("max").astype(np.int64, copy=False)
    out["SrcIPDstPortRange"] = (out["SrcIPDstPortMax"] - out["SrcIPDstPortMin"]).clip(lower=0).astype(
        np.int64, copy=False
    )
    out["SrcIPDstPortRangePerUniq"] = (
        out["SrcIPDstPortRange"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPUniqueDstPort"].astype(np.float32, copy=False), 1.0)
    )
    out["SrcIPUniqueDstIP"] = dst_ip_int.groupby(src_ip_int, sort=False, dropna=False).transform("nunique").astype(
        np.int64, copy=False
    )
    out["SrcIPDstPortUniqRatio"] = (
        out["SrcIPUniqueDstPort"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPFlowCount"].astype(np.float32, copy=False), 1.0)
    )
    out["SrcIPDstIPUniqRatio"] = (
        out["SrcIPUniqueDstIP"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPFlowCount"].astype(np.float32, copy=False), 1.0)
    )

    g_src_min = out.groupby([src_ip_int, out["TsMinuteOfDay"]], sort=False, dropna=False)
    out["SrcIPMinuteFlowCount"] = g_src_min["__row_id"].transform("count").astype(np.int64, copy=False)
    out["SrcIPMinuteUniqueDstPort"] = g_src_min["DstPortInt"].transform("nunique").astype(np.int64, copy=False)
    out["SrcIPMinuteDstPortMin"] = g_src_min["DstPortInt"].transform("min").astype(np.int64, copy=False)
    out["SrcIPMinuteDstPortMax"] = g_src_min["DstPortInt"].transform("max").astype(np.int64, copy=False)
    out["SrcIPMinuteDstPortRange"] = (out["SrcIPMinuteDstPortMax"] - out["SrcIPMinuteDstPortMin"]).clip(
        lower=0
    ).astype(np.int64, copy=False)
    out["SrcIPMinuteDstPortRangePerUniq"] = (
        out["SrcIPMinuteDstPortRange"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPMinuteUniqueDstPort"].astype(np.float32, copy=False), 1.0)
    )
    out["SrcIPMinuteDstPortUniqRatio"] = (
        out["SrcIPMinuteUniqueDstPort"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPMinuteFlowCount"].astype(np.float32, copy=False), 1.0)
    )

    g_src_hour = out.groupby([src_ip_int, out["TsDayOfWeek"], out["TsHour"]], sort=False, dropna=False)
    out["SrcIPHourFlowCount"] = g_src_hour["__row_id"].transform("count").astype(np.int64, copy=False)
    out["SrcIPHourUniqueDstPort"] = g_src_hour["DstPortInt"].transform("nunique").astype(np.int64, copy=False)
    out["SrcIPHourDstPortMin"] = g_src_hour["DstPortInt"].transform("min").astype(np.int64, copy=False)
    out["SrcIPHourDstPortMax"] = g_src_hour["DstPortInt"].transform("max").astype(np.int64, copy=False)
    out["SrcIPHourDstPortRange"] = (out["SrcIPHourDstPortMax"] - out["SrcIPHourDstPortMin"]).clip(lower=0).astype(
        np.int64, copy=False
    )
    out["SrcIPHourDstPortRangePerUniq"] = (
        out["SrcIPHourDstPortRange"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPHourUniqueDstPort"].astype(np.float32, copy=False), 1.0)
    )
    out["SrcIPHourDstPortUniqRatio"] = (
        out["SrcIPHourUniqueDstPort"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPHourFlowCount"].astype(np.float32, copy=False), 1.0)
    )

    ts_minute = out["TsMinuteOfDay"].astype(np.int64, copy=False)
    out["Ts5MinOfDay"] = (ts_minute // 5).astype(np.int64, copy=False)
    g_src_5m = out.groupby([src_ip_int, out["Ts5MinOfDay"]], sort=False, dropna=False)
    g_dst_5m = out.groupby([dst_ip_int, out["Ts5MinOfDay"]], sort=False, dropna=False)
    g_pair_5m = out.groupby([src_ip_int, dst_ip_int, out["Ts5MinOfDay"]], sort=False, dropna=False)
    f5m = {
        "SrcIP5MinFlowCount": g_src_5m["__row_id"].transform("count").astype(np.int64, copy=False),
        "SrcIP5MinUniqueDstPort": g_src_5m["DstPortInt"].transform("nunique").astype(np.int64, copy=False),
        "DstIP5MinFlowCount": g_dst_5m["__row_id"].transform("count").astype(np.int64, copy=False),
        "DstIP5MinUniqueSrcIP": src_ip_int.groupby([dst_ip_int, out["Ts5MinOfDay"]], sort=False, dropna=False)
        .transform("nunique")
        .astype(np.int64, copy=False),
        "SrcDstPair5MinFlowCount": g_pair_5m["__row_id"].transform("count").astype(np.int64, copy=False),
    }
    f5m["SrcIP5MinDstPortUniqRatio"] = (
        f5m["SrcIP5MinUniqueDstPort"].astype(np.float32, copy=False)
        / np.maximum(f5m["SrcIP5MinFlowCount"].astype(np.float32, copy=False), 1.0)
    )
    f5m["DstIP5MinSrcIPUniqRatio"] = (
        f5m["DstIP5MinUniqueSrcIP"].astype(np.float32, copy=False)
        / np.maximum(f5m["DstIP5MinFlowCount"].astype(np.float32, copy=False), 1.0)
    )
    out = pd.concat([out, pd.DataFrame(f5m, index=out.index)], axis=1)

    out["Ts15MinOfDay"] = (ts_minute // 15).astype(np.int64, copy=False)
    g_src_15m = out.groupby([src_ip_int, out["Ts15MinOfDay"]], sort=False, dropna=False)
    g_dst_15m = out.groupby([dst_ip_int, out["Ts15MinOfDay"]], sort=False, dropna=False)
    g_pair_15m = out.groupby([src_ip_int, dst_ip_int, out["Ts15MinOfDay"]], sort=False, dropna=False)
    f15m = {
        "SrcIP15MinFlowCount": g_src_15m["__row_id"].transform("count").astype(np.int64, copy=False),
        "SrcIP15MinUniqueDstPort": g_src_15m["DstPortInt"].transform("nunique").astype(np.int64, copy=False),
        "DstIP15MinFlowCount": g_dst_15m["__row_id"].transform("count").astype(np.int64, copy=False),
        "DstIP15MinUniqueSrcIP": src_ip_int.groupby([dst_ip_int, out["Ts15MinOfDay"]], sort=False, dropna=False)
        .transform("nunique")
        .astype(np.int64, copy=False),
        "SrcDstPair15MinFlowCount": g_pair_15m["__row_id"].transform("count").astype(np.int64, copy=False),
    }
    f15m["SrcIP15MinDstPortUniqRatio"] = (
        f15m["SrcIP15MinUniqueDstPort"].astype(np.float32, copy=False)
        / np.maximum(f15m["SrcIP15MinFlowCount"].astype(np.float32, copy=False), 1.0)
    )
    f15m["DstIP15MinSrcIPUniqRatio"] = (
        f15m["DstIP15MinUniqueSrcIP"].astype(np.float32, copy=False)
        / np.maximum(f15m["DstIP15MinFlowCount"].astype(np.float32, copy=False), 1.0)
    )
    out = pd.concat([out, pd.DataFrame(f15m, index=out.index)], axis=1)

    g_pair = out.groupby([src_ip_int, dst_ip_int], sort=False, dropna=False)
    out["SrcDstPairFlowCount"] = g_pair["__row_id"].transform("count").astype(np.int64, copy=False)

    g_dst = out.groupby(dst_ip_int, sort=False, dropna=False)
    out["DstIPFlowCount"] = g_dst["__row_id"].transform("count").astype(np.int64, copy=False)
    out["DstIPUniqueSrcIP"] = src_ip_int.groupby(dst_ip_int, sort=False, dropna=False).transform("nunique").astype(
        np.int64, copy=False
    )
    out["DstIPSrcIPUniqRatio"] = (
        out["DstIPUniqueSrcIP"].astype(np.float32, copy=False)
        / np.maximum(out["DstIPFlowCount"].astype(np.float32, copy=False), 1.0)
    )

    g_dport = out.groupby(out["DstPortInt"], sort=False, dropna=False)
    out["DstPortFlowCount"] = g_dport["__row_id"].transform("count").astype(np.int64, copy=False)
    out["DstPortUniqueSrcIP"] = src_ip_int.groupby(out["DstPortInt"], sort=False, dropna=False).transform("nunique").astype(
        np.int64, copy=False
    )
    out["DstPortSrcIPUniqRatio"] = (
        out["DstPortUniqueSrcIP"].astype(np.float32, copy=False)
        / np.maximum(out["DstPortFlowCount"].astype(np.float32, copy=False), 1.0)
    )

    out["SrcPortWellKnown"] = (src_port <= 1023).fillna(False).astype(np.int64)
    out["DstPortWellKnown"] = (dst_port <= 1023).fillna(False).astype(np.int64)
    out["DstPortWeb"] = dst_port.isin([80, 443, 8080]).fillna(False).astype(np.int64)
    out["DstPortSSH"] = dst_port.isin([22]).fillna(False).astype(np.int64)
    out["DstPortDB"] = dst_port.isin([3306, 5432, 1433]).fillna(False).astype(np.int64)
    out["DstPort9000"] = dst_port.isin([9000]).fillna(False).astype(np.int64)

    out["FlowDirInternal"] = ((out["SrcIsPrivate"] == 1) & (out["DstIsPrivate"] == 1)).astype(np.int64)
    out["FlowDirOutbound"] = ((out["SrcIsPrivate"] == 1) & (out["DstIsPrivate"] == 0)).astype(np.int64)
    out["FlowDirInbound"] = ((out["SrcIsPrivate"] == 0) & (out["DstIsPrivate"] == 1)).astype(np.int64)
    return out


def add_network_context_group_stats_train_only(
    df: pd.DataFrame,
    idx_train: np.ndarray,
) -> pd.DataFrame:
    out = df.copy()
    if "__row_id" not in out.columns:
        out["__row_id"] = np.arange(len(out), dtype=np.int64)

    if "SrcIPInt" not in out.columns or "DstIPInt" not in out.columns:
        raise ValueError("add_network_context_group_stats_train_only requires SrcIPInt and DstIPInt columns.")

    if "DstPortInt" not in out.columns:
        if "Dst Port" in out.columns:
            out["DstPortInt"] = pd.to_numeric(out["Dst Port"], errors="coerce").fillna(-1).astype(np.int64, copy=False)
        elif "Dst Port" in out.columns:
            out["DstPortInt"] = pd.to_numeric(out["Dst Port"], errors="coerce").fillna(-1).astype(np.int64, copy=False)
        else:
            raise ValueError("add_network_context_group_stats_train_only requires DstPortInt or Dst Port column.")

    if "TsMinuteOfDay" not in out.columns or "TsDayOfWeek" not in out.columns or "TsHour" not in out.columns:
        if "Timestamp" not in out.columns:
            raise ValueError("add_network_context_group_stats_train_only requires Ts* columns or Timestamp column.")
        ts = pd.to_datetime(out["Timestamp"].astype(str), errors="coerce", dayfirst=True)
        out["TsHour"] = ts.dt.hour.fillna(-1).astype(np.int64)
        out["TsDayOfWeek"] = ts.dt.dayofweek.fillna(-1).astype(np.int64)
        out["TsMinuteOfDay"] = (ts.dt.hour.fillna(0) * 60 + ts.dt.minute.fillna(0)).astype(np.int64)

    src_ip_int = out["SrcIPInt"].fillna(-1).astype(np.int64, copy=False)
    dst_ip_int = out["DstIPInt"].fillna(-1).astype(np.int64, copy=False)
    dst_port_int = out["DstPortInt"].fillna(-1).astype(np.int64, copy=False)
    ts_minute = out["TsMinuteOfDay"].fillna(-1).astype(np.int64, copy=False)
    ts_dow = out["TsDayOfWeek"].fillna(-1).astype(np.int64, copy=False)
    ts_hour = out["TsHour"].fillna(-1).astype(np.int64, copy=False)

    if "Ts5MinOfDay" not in out.columns:
        out["Ts5MinOfDay"] = (ts_minute // 5).astype(np.int64, copy=False)
    if "Ts15MinOfDay" not in out.columns:
        out["Ts15MinOfDay"] = (ts_minute // 15).astype(np.int64, copy=False)
    ts_5m = out["Ts5MinOfDay"].fillna(-1).astype(np.int64, copy=False)
    ts_15m = out["Ts15MinOfDay"].fillna(-1).astype(np.int64, copy=False)

    train = out.iloc[idx_train]
    tr_src = train["SrcIPInt"].fillna(-1).astype(np.int64, copy=False)
    tr_dst = train["DstIPInt"].fillna(-1).astype(np.int64, copy=False)
    tr_dport = train["DstPortInt"].fillna(-1).astype(np.int64, copy=False)
    tr_min = train["TsMinuteOfDay"].fillna(-1).astype(np.int64, copy=False)
    tr_dow = train["TsDayOfWeek"].fillna(-1).astype(np.int64, copy=False)
    tr_hour = train["TsHour"].fillna(-1).astype(np.int64, copy=False)
    tr_5m = train["Ts5MinOfDay"].fillna(-1).astype(np.int64, copy=False)
    tr_15m = train["Ts15MinOfDay"].fillna(-1).astype(np.int64, copy=False)

    g_src = train.groupby(tr_src, sort=False, dropna=False)
    src_count = g_src["__row_id"].size()
    src_uniq_dport = g_src["DstPortInt"].nunique()
    src_min_dport = g_src["DstPortInt"].min()
    src_max_dport = g_src["DstPortInt"].max()
    src_uniq_dst = pd.Series(tr_dst).groupby(tr_src, sort=False, dropna=False).nunique()

    out["SrcIPFlowCount"] = pd.Series(src_ip_int, index=out.index).map(src_count).fillna(0).astype(np.int64)
    out["SrcIPUniqueDstPort"] = pd.Series(src_ip_int, index=out.index).map(src_uniq_dport).fillna(0).astype(np.int64)
    out["SrcIPDstPortMin"] = pd.Series(src_ip_int, index=out.index).map(src_min_dport).fillna(-1).astype(np.int64)
    out["SrcIPDstPortMax"] = pd.Series(src_ip_int, index=out.index).map(src_max_dport).fillna(-1).astype(np.int64)
    out["SrcIPDstPortRange"] = (out["SrcIPDstPortMax"] - out["SrcIPDstPortMin"]).clip(lower=0).astype(np.int64)
    out["SrcIPUniqueDstIP"] = pd.Series(src_ip_int, index=out.index).map(src_uniq_dst).fillna(0).astype(np.int64)
    out["SrcIPDstPortRangePerUniq"] = (
        out["SrcIPDstPortRange"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPUniqueDstPort"].astype(np.float32, copy=False), 1.0)
    )
    out["SrcIPDstPortUniqRatio"] = (
        out["SrcIPUniqueDstPort"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPFlowCount"].astype(np.float32, copy=False), 1.0)
    )
    out["SrcIPDstIPUniqRatio"] = (
        out["SrcIPUniqueDstIP"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPFlowCount"].astype(np.float32, copy=False), 1.0)
    )

    src_min_stats = pd.DataFrame({"SrcIPInt": tr_src, "TsMinuteOfDay": tr_min, "DstPortInt": tr_dport}).groupby(
        ["SrcIPInt", "TsMinuteOfDay"], sort=False, dropna=False
    )["DstPortInt"].agg(["size", "nunique", "min", "max"])
    mi_full = pd.MultiIndex.from_arrays([src_ip_int, ts_minute], names=["SrcIPInt", "TsMinuteOfDay"])
    out["SrcIPMinuteFlowCount"] = src_min_stats["size"].reindex(mi_full).fillna(0).astype(np.int64).to_numpy()
    out["SrcIPMinuteUniqueDstPort"] = src_min_stats["nunique"].reindex(mi_full).fillna(0).astype(np.int64).to_numpy()
    out["SrcIPMinuteDstPortMin"] = src_min_stats["min"].reindex(mi_full).fillna(-1).astype(np.int64).to_numpy()
    out["SrcIPMinuteDstPortMax"] = src_min_stats["max"].reindex(mi_full).fillna(-1).astype(np.int64).to_numpy()
    out["SrcIPMinuteDstPortRange"] = (out["SrcIPMinuteDstPortMax"] - out["SrcIPMinuteDstPortMin"]).clip(lower=0).astype(
        np.int64
    )
    out["SrcIPMinuteDstPortRangePerUniq"] = (
        out["SrcIPMinuteDstPortRange"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPMinuteUniqueDstPort"].astype(np.float32, copy=False), 1.0)
    )
    out["SrcIPMinuteDstPortUniqRatio"] = (
        out["SrcIPMinuteUniqueDstPort"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPMinuteFlowCount"].astype(np.float32, copy=False), 1.0)
    )

    src_hour_stats = pd.DataFrame(
        {"SrcIPInt": tr_src, "TsDayOfWeek": tr_dow, "TsHour": tr_hour, "DstPortInt": tr_dport}
    ).groupby(["SrcIPInt", "TsDayOfWeek", "TsHour"], sort=False, dropna=False)["DstPortInt"].agg(["size", "nunique", "min", "max"])
    mi_hour_full = pd.MultiIndex.from_arrays([src_ip_int, ts_dow, ts_hour], names=["SrcIPInt", "TsDayOfWeek", "TsHour"])
    out["SrcIPHourFlowCount"] = src_hour_stats["size"].reindex(mi_hour_full).fillna(0).astype(np.int64).to_numpy()
    out["SrcIPHourUniqueDstPort"] = src_hour_stats["nunique"].reindex(mi_hour_full).fillna(0).astype(np.int64).to_numpy()
    out["SrcIPHourDstPortMin"] = src_hour_stats["min"].reindex(mi_hour_full).fillna(-1).astype(np.int64).to_numpy()
    out["SrcIPHourDstPortMax"] = src_hour_stats["max"].reindex(mi_hour_full).fillna(-1).astype(np.int64).to_numpy()
    out["SrcIPHourDstPortRange"] = (out["SrcIPHourDstPortMax"] - out["SrcIPHourDstPortMin"]).clip(lower=0).astype(np.int64)
    out["SrcIPHourDstPortRangePerUniq"] = (
        out["SrcIPHourDstPortRange"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPHourUniqueDstPort"].astype(np.float32, copy=False), 1.0)
    )
    out["SrcIPHourDstPortUniqRatio"] = (
        out["SrcIPHourUniqueDstPort"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIPHourFlowCount"].astype(np.float32, copy=False), 1.0)
    )

    src_5m_stats = pd.DataFrame({"SrcIPInt": tr_src, "Ts5MinOfDay": tr_5m, "DstPortInt": tr_dport}).groupby(
        ["SrcIPInt", "Ts5MinOfDay"], sort=False, dropna=False
    )["DstPortInt"].agg(["size", "nunique"])
    dst_5m_flow = pd.DataFrame({"DstIPInt": tr_dst, "Ts5MinOfDay": tr_5m}).groupby(
        ["DstIPInt", "Ts5MinOfDay"], sort=False, dropna=False
    ).size()
    dst_5m_uniq_src = pd.DataFrame({"DstIPInt": tr_dst, "Ts5MinOfDay": tr_5m, "SrcIPInt": tr_src}).groupby(
        ["DstIPInt", "Ts5MinOfDay"], sort=False, dropna=False
    )["SrcIPInt"].nunique()
    pair_5m_flow = pd.DataFrame({"SrcIPInt": tr_src, "DstIPInt": tr_dst, "Ts5MinOfDay": tr_5m}).groupby(
        ["SrcIPInt", "DstIPInt", "Ts5MinOfDay"], sort=False, dropna=False
    ).size()

    mi_src_5m = pd.MultiIndex.from_arrays([src_ip_int, ts_5m], names=["SrcIPInt", "Ts5MinOfDay"])
    mi_dst_5m = pd.MultiIndex.from_arrays([dst_ip_int, ts_5m], names=["DstIPInt", "Ts5MinOfDay"])
    mi_pair_5m = pd.MultiIndex.from_arrays([src_ip_int, dst_ip_int, ts_5m], names=["SrcIPInt", "DstIPInt", "Ts5MinOfDay"])
    out["SrcIP5MinFlowCount"] = src_5m_stats["size"].reindex(mi_src_5m).fillna(0).astype(np.int64).to_numpy()
    out["SrcIP5MinUniqueDstPort"] = src_5m_stats["nunique"].reindex(mi_src_5m).fillna(0).astype(np.int64).to_numpy()
    out["DstIP5MinFlowCount"] = dst_5m_flow.reindex(mi_dst_5m).fillna(0).astype(np.int64).to_numpy()
    out["DstIP5MinUniqueSrcIP"] = dst_5m_uniq_src.reindex(mi_dst_5m).fillna(0).astype(np.int64).to_numpy()
    out["SrcDstPair5MinFlowCount"] = pair_5m_flow.reindex(mi_pair_5m).fillna(0).astype(np.int64).to_numpy()
    out["SrcIP5MinDstPortUniqRatio"] = (
        out["SrcIP5MinUniqueDstPort"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIP5MinFlowCount"].astype(np.float32, copy=False), 1.0)
    )
    out["DstIP5MinSrcIPUniqRatio"] = (
        out["DstIP5MinUniqueSrcIP"].astype(np.float32, copy=False)
        / np.maximum(out["DstIP5MinFlowCount"].astype(np.float32, copy=False), 1.0)
    )

    src_15m_stats = pd.DataFrame({"SrcIPInt": tr_src, "Ts15MinOfDay": tr_15m, "DstPortInt": tr_dport}).groupby(
        ["SrcIPInt", "Ts15MinOfDay"], sort=False, dropna=False
    )["DstPortInt"].agg(["size", "nunique"])
    dst_15m_flow = pd.DataFrame({"DstIPInt": tr_dst, "Ts15MinOfDay": tr_15m}).groupby(
        ["DstIPInt", "Ts15MinOfDay"], sort=False, dropna=False
    ).size()
    dst_15m_uniq_src = pd.DataFrame({"DstIPInt": tr_dst, "Ts15MinOfDay": tr_15m, "SrcIPInt": tr_src}).groupby(
        ["DstIPInt", "Ts15MinOfDay"], sort=False, dropna=False
    )["SrcIPInt"].nunique()
    pair_15m_flow = pd.DataFrame({"SrcIPInt": tr_src, "DstIPInt": tr_dst, "Ts15MinOfDay": tr_15m}).groupby(
        ["SrcIPInt", "DstIPInt", "Ts15MinOfDay"], sort=False, dropna=False
    ).size()

    mi_src_15m = pd.MultiIndex.from_arrays([src_ip_int, ts_15m], names=["SrcIPInt", "Ts15MinOfDay"])
    mi_dst_15m = pd.MultiIndex.from_arrays([dst_ip_int, ts_15m], names=["DstIPInt", "Ts15MinOfDay"])
    mi_pair_15m = pd.MultiIndex.from_arrays(
        [src_ip_int, dst_ip_int, ts_15m], names=["SrcIPInt", "DstIPInt", "Ts15MinOfDay"]
    )
    out["SrcIP15MinFlowCount"] = src_15m_stats["size"].reindex(mi_src_15m).fillna(0).astype(np.int64).to_numpy()
    out["SrcIP15MinUniqueDstPort"] = src_15m_stats["nunique"].reindex(mi_src_15m).fillna(0).astype(np.int64).to_numpy()
    out["DstIP15MinFlowCount"] = dst_15m_flow.reindex(mi_dst_15m).fillna(0).astype(np.int64).to_numpy()
    out["DstIP15MinUniqueSrcIP"] = dst_15m_uniq_src.reindex(mi_dst_15m).fillna(0).astype(np.int64).to_numpy()
    out["SrcDstPair15MinFlowCount"] = pair_15m_flow.reindex(mi_pair_15m).fillna(0).astype(np.int64).to_numpy()
    out["SrcIP15MinDstPortUniqRatio"] = (
        out["SrcIP15MinUniqueDstPort"].astype(np.float32, copy=False)
        / np.maximum(out["SrcIP15MinFlowCount"].astype(np.float32, copy=False), 1.0)
    )
    out["DstIP15MinSrcIPUniqRatio"] = (
        out["DstIP15MinUniqueSrcIP"].astype(np.float32, copy=False)
        / np.maximum(out["DstIP15MinFlowCount"].astype(np.float32, copy=False), 1.0)
    )

    pair_flow = pd.DataFrame({"SrcIPInt": tr_src, "DstIPInt": tr_dst}).groupby(["SrcIPInt", "DstIPInt"], sort=False, dropna=False).size()
    mi_pair = pd.MultiIndex.from_arrays([src_ip_int, dst_ip_int], names=["SrcIPInt", "DstIPInt"])
    out["SrcDstPairFlowCount"] = pair_flow.reindex(mi_pair).fillna(0).astype(np.int64).to_numpy()

    dst_flow = pd.Series(tr_dst).value_counts()
    dst_uniq_src = pd.Series(tr_src).groupby(tr_dst, sort=False, dropna=False).nunique()
    out["DstIPFlowCount"] = pd.Series(dst_ip_int, index=out.index).map(dst_flow).fillna(0).astype(np.int64)
    out["DstIPUniqueSrcIP"] = pd.Series(dst_ip_int, index=out.index).map(dst_uniq_src).fillna(0).astype(np.int64)
    out["DstIPSrcIPUniqRatio"] = (
        out["DstIPUniqueSrcIP"].astype(np.float32, copy=False)
        / np.maximum(out["DstIPFlowCount"].astype(np.float32, copy=False), 1.0)
    )

    dport_flow = pd.Series(tr_dport).value_counts()
    dport_uniq_src = pd.Series(tr_src).groupby(tr_dport, sort=False, dropna=False).nunique()
    out["DstPortFlowCount"] = pd.Series(dst_port_int, index=out.index).map(dport_flow).fillna(0).astype(np.int64)
    out["DstPortUniqueSrcIP"] = pd.Series(dst_port_int, index=out.index).map(dport_uniq_src).fillna(0).astype(np.int64)
    out["DstPortSrcIPUniqRatio"] = (
        out["DstPortUniqueSrcIP"].astype(np.float32, copy=False)
        / np.maximum(out["DstPortFlowCount"].astype(np.float32, copy=False), 1.0)
    )

    return out


def load_zapt_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["__source_file"] = os.path.basename(csv_path)
    df["__row_id"] = np.arange(len(df), dtype=np.int64)

    if "label_sub" in df.columns:
        stage = df["label_sub"].astype(str)
    elif "label" in df.columns:
        stage = df["label"].astype(str)
    else:
        raise ValueError("ZAPT dataset requires 'label_sub' or 'label' column.")

    df["Stage"] = stage.where(stage != "Benign", other="Benign")
    df["Activity"] = np.where(df["Stage"].astype(str) == "Benign", "Normal", stage).astype(str)

    if {"src_ip_x", "dst_ip_x", "sport", "dport", "start_time"}.issubset(df.columns):
        df["Src IP"] = df["src_ip_x"]
        df["Dst IP"] = df["dst_ip_x"]
        df["Src Port"] = df["sport"]
        df["Dst Port"] = df["dport"]
        df["Timestamp"] = df["start_time"]
        df = _add_network_context_features(
            df=df,
            src_ip_col="Src IP",
            dst_ip_col="Dst IP",
            src_port_col="Src Port",
            dst_port_col="Dst Port",
            timestamp_col="Timestamp",
        )

    return df


def _build_cic2024_derived_feature_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()

    def _num(col: str) -> pd.Series | None:
        if col not in out.columns:
            return None
        return pd.to_numeric(out[col], errors="coerce")

    new_cols: dict[str, pd.Series] = {}

    total_fwd_pkt = _num("Total Fwd Packet")
    total_bwd_pkt = _num("Total Bwd packets")
    total_fwd_len = _num("Total Length of Fwd Packet")
    total_bwd_len = _num("Total Length of Bwd Packet")
    pkt_len_mean = _num("Packet Length Mean")
    pkt_len_std = _num("Packet Length Std")
    flow_iat_mean = _num("Flow IAT Mean")
    flow_iat_std = _num("Flow IAT Std")
    fwd_iat_mean = _num("Fwd IAT Mean")
    bwd_iat_mean = _num("Bwd IAT Mean")
    flow_bytes = _num("Flow Bytes/s")
    flow_pkts = _num("Flow Packets/s")
    src_flow = _num("SrcIPFlowCount")
    dst_flow = _num("DstIPFlowCount")
    pair_flow = _num("SrcDstPairFlowCount")
    dst_port_flow = _num("DstPortFlowCount")
    src_min_flow = _num("SrcIPMinuteFlowCount")
    src_hour_flow = _num("SrcIPHourFlowCount")
    src_5m_flow = _num("SrcIP5MinFlowCount")
    src_15m_flow = _num("SrcIP15MinFlowCount")
    dst_5m_flow = _num("DstIP5MinFlowCount")
    dst_15m_flow = _num("DstIP15MinFlowCount")
    src_uniq_port = _num("SrcIPUniqueDstPort")
    src_min_uniq_port = _num("SrcIPMinuteUniqueDstPort")
    src_hour_uniq_port = _num("SrcIPHourUniqueDstPort")
    src_5m_uniq_port = _num("SrcIP5MinUniqueDstPort")
    src_15m_uniq_port = _num("SrcIP15MinUniqueDstPort")
    dst_uniq_src = _num("DstIPUniqueSrcIP")
    dst_5m_uniq_src = _num("DstIP5MinUniqueSrcIP")
    dst_15m_uniq_src = _num("DstIP15MinUniqueSrcIP")

    if total_fwd_pkt is not None and total_bwd_pkt is not None:
        pkt_total = (total_fwd_pkt.fillna(0) + total_bwd_pkt.fillna(0)).astype(np.float32, copy=False)
        new_cols["FlowPktTotalDerived"] = pkt_total
        new_cols["FlowPktImbalance"] = _safe_ratio(total_fwd_pkt - total_bwd_pkt, pkt_total + 1.0)
    if total_fwd_len is not None and total_bwd_len is not None:
        byte_total = (total_fwd_len.fillna(0) + total_bwd_len.fillna(0)).astype(np.float32, copy=False)
        new_cols["FlowByteTotalDerived"] = byte_total
        new_cols["FlowByteImbalance"] = _safe_ratio(total_fwd_len - total_bwd_len, byte_total + 1.0)
    if total_fwd_len is not None and total_fwd_pkt is not None:
        new_cols["FwdAvgPktLenDerived"] = _safe_ratio(total_fwd_len, total_fwd_pkt + 1.0)
    if total_bwd_len is not None and total_bwd_pkt is not None:
        new_cols["BwdAvgPktLenDerived"] = _safe_ratio(total_bwd_len, total_bwd_pkt + 1.0)
    if pkt_len_mean is not None and pkt_len_std is not None:
        new_cols["PktLenCvDerived"] = _safe_ratio(pkt_len_std, pkt_len_mean + 1.0)
    if flow_iat_mean is not None and flow_iat_std is not None:
        new_cols["FlowIATCvDerived"] = _safe_ratio(flow_iat_std, flow_iat_mean + 1.0)
    if fwd_iat_mean is not None and bwd_iat_mean is not None:
        new_cols["FwdBwdIATRatio"] = _safe_ratio(fwd_iat_mean + 1.0, bwd_iat_mean + 1.0)
    if flow_bytes is not None and flow_pkts is not None:
        new_cols["BytesPerPacketDerived"] = _safe_ratio(flow_bytes + 1.0, flow_pkts + 1.0)
    if src_flow is not None and dst_flow is not None:
        new_cols["SrcDstFlowCountRatio"] = _safe_ratio(src_flow + 1.0, dst_flow + 1.0)
    if pair_flow is not None and src_flow is not None:
        new_cols["PairSrcFlowRatio"] = _safe_ratio(pair_flow + 1.0, src_flow + 1.0)
    if pair_flow is not None and dst_flow is not None:
        new_cols["PairDstFlowRatio"] = _safe_ratio(pair_flow + 1.0, dst_flow + 1.0)
    if dst_port_flow is not None and src_flow is not None:
        new_cols["DstPortVsSrcFlowRatio"] = _safe_ratio(dst_port_flow + 1.0, src_flow + 1.0)
    if src_min_flow is not None and src_hour_flow is not None:
        new_cols["BurstMinuteHourRatio"] = _safe_ratio(src_min_flow + 1.0, src_hour_flow + 1.0)
    if src_5m_flow is not None and src_15m_flow is not None:
        new_cols["Burst5m15mRatio"] = _safe_ratio(src_5m_flow + 1.0, src_15m_flow + 1.0)
    if dst_5m_flow is not None and dst_15m_flow is not None:
        new_cols["DstBurst5m15mRatio"] = _safe_ratio(dst_5m_flow + 1.0, dst_15m_flow + 1.0)
    if src_uniq_port is not None and src_flow is not None:
        new_cols["SrcPortDiversityDerived"] = _safe_ratio(src_uniq_port + 1.0, src_flow + 1.0)
    if src_min_uniq_port is not None and src_min_flow is not None:
        new_cols["SrcMinutePortDiversityDerived"] = _safe_ratio(src_min_uniq_port + 1.0, src_min_flow + 1.0)
    if src_hour_uniq_port is not None and src_hour_flow is not None:
        new_cols["SrcHourPortDiversityDerived"] = _safe_ratio(src_hour_uniq_port + 1.0, src_hour_flow + 1.0)
    if src_5m_uniq_port is not None and src_5m_flow is not None:
        new_cols["Src5mPortDiversityDerived"] = _safe_ratio(src_5m_uniq_port + 1.0, src_5m_flow + 1.0)
    if src_15m_uniq_port is not None and src_15m_flow is not None:
        new_cols["Src15mPortDiversityDerived"] = _safe_ratio(src_15m_uniq_port + 1.0, src_15m_flow + 1.0)
    if dst_uniq_src is not None and dst_flow is not None:
        new_cols["DstSrcDiversityDerived"] = _safe_ratio(dst_uniq_src + 1.0, dst_flow + 1.0)
    if dst_5m_uniq_src is not None and dst_5m_flow is not None:
        new_cols["Dst5mSrcDiversityDerived"] = _safe_ratio(dst_5m_uniq_src + 1.0, dst_5m_flow + 1.0)
    if dst_15m_uniq_src is not None and dst_15m_flow is not None:
        new_cols["Dst15mSrcDiversityDerived"] = _safe_ratio(dst_15m_uniq_src + 1.0, dst_15m_flow + 1.0)

    if not new_cols:
        return (
            pd.DataFrame(index=out.index),
            pd.DataFrame(index=out.index),
        )

    derived_df = pd.DataFrame(new_cols, index=out.index)
    log_cols: dict[str, pd.Series] = {}
    for col in derived_df.columns:
        v = pd.to_numeric(derived_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        log_cols[f"Log1p_{col}"] = np.log1p(v.clip(lower=0)).astype(np.float32)
    log_df = pd.DataFrame(log_cols, index=out.index)
    return derived_df, log_df


def _add_cic2024_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    derived_df, log_df = _build_cic2024_derived_feature_frames(out)
    if derived_df.empty and log_df.empty:
        return out
    return pd.concat([out, derived_df, log_df], axis=1)


def refresh_cic2024_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    derived_df, log_df = _build_cic2024_derived_feature_frames(out)
    for col in list(derived_df.columns) + list(log_df.columns):
        out[col] = derived_df[col] if col in derived_df.columns else log_df[col]
    return out


def load_cic2024_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["__source_file"] = os.path.basename(csv_path)
    df["__row_id"] = np.arange(len(df), dtype=np.int64)

    if "Activity" not in df.columns:
        raise ValueError("CIC-2024 dataset requires 'Activity' column.")

    act = df["Activity"].astype(str)
    is_benign = act.isin(["Normal", "Benign", "BENIGN"])
    df["Stage"] = np.where(is_benign, "Benign", act).astype(str)
    df["Activity"] = np.where(is_benign, "Normal", act).astype(str)

    if {"Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp"}.issubset(df.columns):
        df = _add_network_context_features(
            df=df,
            src_ip_col="Src IP",
            dst_ip_col="Dst IP",
            src_port_col="Src Port",
            dst_port_col="Dst Port",
            timestamp_col="Timestamp",
        )

        src_port = pd.to_numeric(df["Src Port"], errors="coerce")
        dst_port = pd.to_numeric(df["Dst Port"], errors="coerce")
        extra_cols = pd.DataFrame(
            {
                "SameSubnet16": (
                    (df["SrcIsPrivate"] == 1)
                    & (df["DstIsPrivate"] == 1)
                    & (df["SrcIPInt"].fillna(-1).astype(np.int64) // 65536 == df["DstIPInt"].fillna(-2).astype(np.int64) // 65536)
                ).astype(np.int64),
                "SrcPortEphemeral": (src_port >= 49152).fillna(False).astype(np.int64),
                "DstPortEphemeral": (dst_port >= 49152).fillna(False).astype(np.int64),
                "DstPortDNS": dst_port.isin([53]).fillna(False).astype(np.int64),
                "DstPortNTP": dst_port.isin([123]).fillna(False).astype(np.int64),
                "DstPortRDP": dst_port.isin([3389]).fillna(False).astype(np.int64),
                "DstPortSMB": dst_port.isin([445]).fillna(False).astype(np.int64),
                "DstPortMail": dst_port.isin([25, 110, 143, 587, 993, 995]).fillna(False).astype(np.int64),
            },
            index=df.index,
        )
        df = pd.concat([df, extra_cols], axis=1)

    df = _add_cic2024_derived_features(df)

    log_targets = [
        "Flow Duration",
        "Total Fwd Packet",
        "Total Bwd packets",
        "Total Length of Fwd Packet",
        "Total Length of Bwd Packet",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Fwd Packets/s",
        "Bwd Packets/s",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Total",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Total",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Packet Length Min",
        "Packet Length Max",
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "Active Mean",
        "Active Std",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min",
        "SrcIPFlowCount",
        "SrcIPUniqueDstPort",
        "SrcIPUniqueDstIP",
        "SrcIPDstPortRange",
        "SrcIPMinuteFlowCount",
        "SrcIPMinuteUniqueDstPort",
        "SrcIPMinuteDstPortRange",
        "SrcIPHourFlowCount",
        "SrcIPHourUniqueDstPort",
        "SrcIPHourDstPortRange",
        "SrcDstPairFlowCount",
        "DstIPFlowCount",
        "DstIPUniqueSrcIP",
        "DstPortFlowCount",
        "DstPortUniqueSrcIP",
    ]
    log_cols: dict[str, pd.Series] = {}
    for c in log_targets:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").clip(lower=0)
        log_cols[f"Log1p_{c}"] = np.log1p(v).astype(np.float32)
    if log_cols:
        df = pd.concat([df, pd.DataFrame(log_cols, index=df.index)], axis=1)
    return df


def load_earlycrow_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["__source_file"] = os.path.basename(csv_path)
    df["__row_id"] = np.arange(len(df), dtype=np.int64)

    if "label" in df.columns:
        lab = df["label"].astype(str).str.lower()
        is_benign = lab.isin(["benign", "legitimate", "normal"])
    elif "multiple_label" in df.columns:
        lab = df["multiple_label"].astype(str).str.lower()
        is_benign = lab.isin(["benign", "legitimate", "normal"])
    else:
        raise ValueError("EarlyCrow dataset requires 'label' or 'multiple_label' column.")

    if "multiple_label" in df.columns:
        stage_raw = df["multiple_label"].astype(str)
        stage_raw = stage_raw.where(~stage_raw.astype(str).str.lower().isin(["benign", "legitimate", "normal"]), other="Benign")
    else:
        stage_raw = np.where(is_benign, "Benign", "Malicious").astype(str)

    df["Stage"] = np.where(is_benign, "Benign", stage_raw).astype(str)
    df["Activity"] = np.where(df["Stage"].astype(str) == "Benign", "Normal", df["Stage"].astype(str)).astype(str)

    if {"Source", "Destination"}.issubset(df.columns):
        df["Src IP"] = df["Source"]
        df["Dst IP"] = df["Destination"]
    if {"tcp_srcport", "udp_srcport"}.issubset(df.columns):
        tcp_s = pd.to_numeric(df["tcp_srcport"], errors="coerce")
        udp_s = pd.to_numeric(df["udp_srcport"], errors="coerce")
        df["Src Port"] = tcp_s.where(tcp_s.notna(), udp_s)
    if {"tcp_dstport", "udp_dstport"}.issubset(df.columns):
        tcp_d = pd.to_numeric(df["tcp_dstport"], errors="coerce")
        udp_d = pd.to_numeric(df["udp_dstport"], errors="coerce")
        df["Dst Port"] = tcp_d.where(tcp_d.notna(), udp_d)
    if "Absolute_Time" in df.columns:
        df["Timestamp"] = df["Absolute_Time"]

    if {"Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp"}.issubset(df.columns):
        df = _add_network_context_features(
            df=df,
            src_ip_col="Src IP",
            dst_ip_col="Dst IP",
            src_port_col="Src Port",
            dst_port_col="Dst Port",
            timestamp_col="Timestamp",
        )

    return df


def _coerce_numeric_features(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        col = out[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        out[c] = pd.to_numeric(col, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _filter_numeric_feature_cols(df: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
    keep: list[str] = []
    for c in candidate_cols:
        col = df[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        s = pd.to_numeric(col, errors="coerce")
        if s.notna().any():
            keep.append(c)
    return keep


def _apply_selected_feature_cols(feature_cols: list[str], selected_feature_cols: list[str] | None) -> list[str]:
    if not selected_feature_cols:
        return feature_cols
    allowed = {str(c) for c in selected_feature_cols}
    picked = [c for c in feature_cols if str(c) in allowed]
    if not picked:
        raise ValueError("Selected feature subset removed all numeric feature columns.")
    return picked


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    idx_train: np.ndarray
    row_id_train: np.ndarray

    X_val: np.ndarray
    y_val: np.ndarray
    idx_val: np.ndarray
    row_id_val: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray
    idx_test: np.ndarray
    row_id_test: np.ndarray

    feature_cols: list[str]
    scaler: StandardScaler


def split_indices(
    y: np.ndarray,
    seed: int,
    test_size: float,
    val_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_all = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=test_size, random_state=seed, stratify=y
    )
    y_trainval = y[idx_trainval]
    val_ratio_in_trainval = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_ratio_in_trainval,
        random_state=seed,
        stratify=y_trainval,
    )
    return idx_train.astype(np.int64), idx_val.astype(np.int64), idx_test.astype(np.int64)


def scale_by_indices(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
) -> SplitData:
    if not feature_cols:
        raise ValueError("No numeric feature columns found after preprocessing.")
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    if "__row_id" in df.columns:
        row_ids = df["__row_id"].to_numpy(dtype=np.int64, copy=False)
    else:
        row_ids = np.arange(len(df), dtype=np.int64)
    col_mean = np.nanmean(X[idx_train], axis=0).astype(np.float32, copy=False)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32, copy=False)

    def _impute(a: np.ndarray) -> np.ndarray:
        if np.isnan(a).any():
            m = np.isnan(a)
            a[m] = col_mean[np.where(m)[1]]
        return a

    scaler = StandardScaler()
    X_train = _impute(X[idx_train]).astype(np.float32, copy=False)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32, copy=False)
    X_val = scaler.transform(_impute(X[idx_val])).astype(np.float32, copy=False)
    X_test = scaler.transform(_impute(X[idx_test])).astype(np.float32, copy=False)
    return SplitData(
        X_train=X_train,
        y_train=y[idx_train].astype(np.int64, copy=False),
        idx_train=idx_train.astype(np.int64, copy=False),
        row_id_train=row_ids[idx_train].astype(np.int64, copy=False),
        X_val=X_val,
        y_val=y[idx_val].astype(np.int64, copy=False),
        idx_val=idx_val.astype(np.int64, copy=False),
        row_id_val=row_ids[idx_val].astype(np.int64, copy=False),
        X_test=X_test,
        y_test=y[idx_test].astype(np.int64, copy=False),
        idx_test=idx_test.astype(np.int64, copy=False),
        row_id_test=row_ids[idx_test].astype(np.int64, copy=False),
        feature_cols=feature_cols,
        scaler=scaler,
    )


def make_stage_task(
    df: pd.DataFrame,
    use_activity_as_stage: bool = False,
    selected_feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    if use_activity_as_stage and "Activity" in df.columns:
        y = (df["Activity"].astype(str) != "Normal").astype(np.int64).to_numpy()
    else:
        y = (df["Stage"].astype(str) != "Benign").astype(np.int64).to_numpy()
    candidate = [c for c in df.columns if c not in META_COLS_DAPT]
    feature_cols = _filter_numeric_feature_cols(df, candidate)
    feature_cols = _apply_selected_feature_cols(feature_cols, selected_feature_cols)
    df_num = _coerce_numeric_features(df, feature_cols)
    if use_activity_as_stage and "Activity" in df.columns:
        y = (df_num["Activity"].astype(str) != "Normal").astype(np.int64).to_numpy()
    else:
        y = (df_num["Stage"].astype(str) != "Benign").astype(np.int64).to_numpy()
    return df_num, y, feature_cols


def make_stage_multiclass_task(
    df: pd.DataFrame,
    selected_feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    candidate = [c for c in df.columns if c not in META_COLS_DAPT]
    feature_cols = _filter_numeric_feature_cols(df, candidate)
    feature_cols = _apply_selected_feature_cols(feature_cols, selected_feature_cols)
    df_num = _coerce_numeric_features(df, feature_cols)
    classes = sorted(df_num["Stage"].astype(str).unique().tolist())
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = df_num["Stage"].astype(str).map(class_to_id).astype(np.int64).to_numpy()
    return df_num, y, feature_cols, classes


def make_activity_task(
    df: pd.DataFrame,
    min_class_count: int = 1,
    selected_feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    malicious = df[df["Stage"].astype(str) != "Benign"].copy()
    malicious = malicious[malicious["Activity"].astype(str) != "Normal"].copy()
    if malicious.columns.duplicated().any():
        malicious = malicious.loc[:, ~malicious.columns.duplicated()].copy()
    candidate = [c for c in malicious.columns if c not in META_COLS_DAPT]
    feature_cols = _filter_numeric_feature_cols(malicious, candidate)
    feature_cols = _apply_selected_feature_cols(feature_cols, selected_feature_cols)
    malicious = _coerce_numeric_features(malicious, feature_cols)

    activity = malicious["Activity"].astype(str)
    if min_class_count > 1:
        counts = activity.value_counts()
        rare = counts[counts < min_class_count].index.tolist()
        if rare:
            activity = activity.where(~activity.isin(rare), other="Other")
            malicious = malicious.copy()
            malicious["Activity"] = activity

    classes = sorted(malicious["Activity"].astype(str).unique().tolist())
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = malicious["Activity"].astype(str).map(class_to_id).astype(np.int64).to_numpy()
    return malicious, y, feature_cols, classes


def split_scale(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    seed: int,
    test_size: float,
    val_size: float,
) -> SplitData:
    idx_train, idx_val, idx_test = split_indices(
        y=y, seed=seed, test_size=test_size, val_size=val_size
    )
    return scale_by_indices(
        df=df,
        y=y,
        feature_cols=feature_cols,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )
