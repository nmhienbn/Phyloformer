# Ignore warnings
import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Rename some PF methods
RENAMER = {
    "PF_gaps+FastME": "PF_Indel+FastME",
    "PF+FastME": "PF_Base+FastME",
    "PF_MRE+FastME": "PF+FastME",
    "PF_cherry+FastME": "PF_Cherry+FastME",
    "PF_pastek+FastME": "PF_SelReg+FastME",
}

# All possible methods -> corresponding color in the 'tab10' palette
METHODS = [
    "IQTree_LG+GC",  # Blue
    "FastTree",  # Orange
    "FastME",  # Green
    "PF+FastME",  # Red
    "PF_Cherry+FastME",  # Purple
    "IQTree_MF",  # Brown
    "PF_SelReg+FastME",  # Pink
    "Hamming+FastME",  # Gray
    "PF_Base+FastME",  # Khaki
    "PF_Indel+FastME",  # Cyan
    "PF_CPU+FastME",
    # "BioNJ",
]

MARKERS = {
    "IQTree_LG+GC": "o",
    "IQTree_MF": "^",
    "FastME": "X",
    "FastTree": "s",
    "PF+FastME": "o",
    "PF_Cherry+FastME": "s",
    "PF_SelReg+FastME": "X",
    "PF_Indel+FastME": "v",
    "PF_Base+FastME": "^",
    "PF_CPU+FastME": "^",
    "Hamming+FastME": "v",
    "PF_QCLOSE+FastME": "D",
}

LINESTYLES = {
    "IQTree_LG+GC": "-",
    "IQTree_MF": "-",
    "FastME": ":",
    "FastTree": "-",
    "PF+FastME": "--",
    "PF_Cherry+FastME": "--",
    "PF_SelReg+FastME": "--",
    "PF_Indel+FastME": "--",
    "PF_Base+FastME": "--",
    "PF_CPU+FastME": ":",
    "Hamming+FastME": ":",
    "PF_QCLOSE+FastME": "-.",
}

LGGC_METHODS = sorted(
    [
        "IQTree_LG+GC",
        "FastTree",
        "FastME",
        "Hamming+FastME",
        "PF+FastME",
    ]
)

LGGC_METHODS_NO_HAMMING = sorted(
    [
        "IQTree_LG+GC",
        "FastTree",
        "FastME",
        "PF+FastME",
    ]
)

LGGC_METHODS_NO_HAMMING_PLUS_QCLOSE = sorted(
    LGGC_METHODS_NO_HAMMING + ["PF_QCLOSE+FastME"]
)

FINE_TUNE_METHODS = sorted(
    [
        "IQTree_MF",
        "FastTree",
        "FastME",
        "PF+FastME",
    ]
)


TIPS_TICKS = [i for i in range(10, 110, 10)]

# Unified styles
STYLES = {
    k: (col, LINESTYLES[k], MARKERS[k])
    for k, col in zip(METHODS, sns.color_palette("tab10", n_colors=len(METHODS)))
}

# Make both IQTree versions have the same color
STYLES["IQTree_MF"] = (
    STYLES["IQTree_LG+GC"][0],
    STYLES["IQTree_MF"][1],
    STYLES["IQTree_MF"][2],
)
# And LG versions of PF
STYLES["PF_CPU+FastME"] = (
    STYLES["PF+FastME"][0],
    STYLES["PF_CPU+FastME"][1],
    STYLES["PF_CPU+FastME"][2],
)
STYLES["PF_Base+FastME"] = (
    STYLES["PF+FastME"][0],
    STYLES["PF_Base+FastME"][1],
    STYLES["PF_Base+FastME"][2],
)
STYLES["PF_QCLOSE+FastME"] = (
    STYLES["PF+FastME"][0],
    LINESTYLES["PF_QCLOSE+FastME"],
    MARKERS["PF_QCLOSE+FastME"],
)


def _register_external_style(method):
    if method in STYLES:
        return
    color = sns.color_palette("tab10", n_colors=len(STYLES) + 1)[-1]
    STYLES[method] = (color, "-.", "D")


def _register_style_alias(method, template_method="PF_QCLOSE+FastME"):
    if method in STYLES:
        return
    if template_method not in STYLES:
        raise ValueError(f"Unknown template method style: {template_method}")
    STYLES[method] = STYLES[template_method]


def _slugify_label(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _methods_with_new_model(label):
    return sorted(LGGC_METHODS_NO_HAMMING + [label])


def _normalize_repeatable_arg(values, default_value):
    if values is None:
        return [default_value]
    if isinstance(values, str):
        return [values]
    cleaned = [v for v in values if v]
    return cleaned if cleaned else [default_value]


def _infer_dataset_key_from_path(path):
    lower = str(path).lower()
    if "final_test_set" in lower:
        return "lggc"
    if "lggc+gaps" in lower or "gaps" in lower:
        return "gaps"
    if "cherry" in lower:
        return "cherry"
    if "pastek" in lower:
        return "pastek"
    raise ValueError(f"Could not infer dataset from path: {path}")


def _load_baseline_topology(dataset_key):
    if dataset_key == "lggc":
        return _load_lggc_topo_500("./data/topos_lggc.csv")

    path_map = {
        "gaps": "./data/topos_gaps.csv",
        "cherry": "./data/topos_cherry.csv",
        "pastek": "./data/topos_pastek.csv",
    }
    dataset_names = {
        "gaps": "Indels",
        "cherry": "Cherry",
        "pastek": "SelReg",
    }
    topo = pd.read_csv(path_map[dataset_key]).copy()
    topo["marker"] = topo["marker"].apply(lambda x: RENAMER.get(x, x))
    topo["dataset"] = dataset_names[dataset_key]
    topo["length"] = 500
    if "n_tips" not in topo.columns:
        topo["n_tips"] = topo["id"].map(_extract_n_tips)
    return topo


def _load_baseline_distances(dataset_key, sample_frac):
    if dataset_key == "lggc":
        dists = pd.read_csv("./data/dists_lggc.csv").sample(
            frac=sample_frac, random_state=42
        )
        dists["n_tips"] = dists["id"].apply(lambda x: x.split("_")[1]).astype(int)
        dists["length"] = dists["id"].apply(lambda x: x.split("_")[-1]).astype(int)
        dists["dataset"] = "LG+GC"
        dists = dists[dists["length"] == 500]
    else:
        path_map = {
            "gaps": "./data/dists_gaps.csv",
            "cherry": "./data/dists_cherry.csv",
            "pastek": "./data/dists_pastek.csv",
        }
        dataset_names = {
            "gaps": "Indels",
            "cherry": "Cherry",
            "pastek": "SelReg",
        }
        dists = pd.read_csv(path_map[dataset_key]).sample(
            frac=sample_frac, random_state=42
        )
        dists["n_tips"] = dists["id"].apply(lambda x: x.split("_")[1]).astype(int)
        dists["length"] = 500
        dists["dataset"] = dataset_names[dataset_key]

    dists["MAE"] = (dists["ref_dist"] - dists["cmp_dist"]).abs()
    dists["MRE"] = dists["MAE"] / dists["ref_dist"]
    dists["marker"] = dists["marker"].apply(lambda x: RENAMER.get(x, x))
    return dists


def _dataset_methods_with_new_model(dataset_key, new_model_label):
    if dataset_key == "lggc":
        return _methods_with_new_model(new_model_label)
    if dataset_key == "cherry":
        return sorted(
            sorted(["IQTree_LG+GC"] + FINE_TUNE_METHODS) + ["PF_Cherry+FastME", new_model_label]
        )
    if dataset_key == "pastek":
        return sorted(
            sorted(["IQTree_LG+GC"] + FINE_TUNE_METHODS) + ["PF_SelReg+FastME", new_model_label]
        )
    if dataset_key == "gaps":
        return sorted(
            [x for x in LGGC_METHODS_NO_HAMMING if x != "PF+FastME"]
            + ["PF_Indel+FastME", new_model_label]
        )
    raise ValueError(f"Unknown dataset key: {dataset_key}")


def _plot_filename_prefix(dataset_key):
    return {
        "lggc": "lggc",
        "gaps": "gaps",
        "cherry": "cherry",
        "pastek": "pastek",
    }[dataset_key]


def _load_new_model_results_by_dataset(
    topo_paths,
    dist_paths,
    new_model_label,
    sample_frac,
    default_length=500,
    skip_missing=False,
):
    results = {}
    for topo_path, dist_path in zip(topo_paths, dist_paths):
        dataset_key = _infer_dataset_key_from_path(topo_path)
        if dataset_key in results:
            raise ValueError(
                f"Duplicate new-model dataset inferred from paths: {dataset_key}"
            )
        if dataset_key != _infer_dataset_key_from_path(dist_path):
            raise ValueError(
                f"Mismatched dataset between topo/dist paths: {topo_path} vs {dist_path}"
            )
        topo_path = Path(topo_path)
        dist_path = Path(dist_path)
        if not topo_path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(f"Missing new-model topo file: {topo_path}")
        if not dist_path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(f"Missing new-model dist file: {dist_path}")
        topo = _load_cmp_topo(topo_path, new_model_label, default_length=default_length)
        dists = _load_cmp_dist_pairwise(
            dist_path,
            new_model_label,
            sample_frac=sample_frac,
            chunksize=1_000_000,
            default_length=default_length,
        )
        dataset_name = {
            "lggc": "LG+GC",
            "gaps": "Indels",
            "cherry": "Cherry",
            "pastek": "SelReg",
        }[dataset_key]
        topo["dataset"] = dataset_name
        dists["dataset"] = dataset_name
        results[dataset_key] = {
            "topo": topo,
            "dists": dists,
            "topo_path": topo_path,
            "dist_path": dist_path,
        }
    return results


# To add titles to legends
class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, **self.text_props)
        handlebox.add_artist(title)
        return title


def _extract_n_tips(tree_id: str) -> int:
    m = re.search(r"_(\d+)_tips(?:_|$)", str(tree_id))
    if m is not None:
        return int(m.group(1))
    parts = str(tree_id).split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    raise ValueError(f"Could not infer n_tips from id='{tree_id}'")


def _extract_length(tree_id: str, default_length=None) -> int:
    tail = str(tree_id).split("_")[-1]
    if tail.isdigit():
        return int(tail)
    if default_length is not None:
        return int(default_length)
    raise ValueError(f"Could not infer length from id='{tree_id}'")


def _parse_labelled_paths(items, kind):
    out = {}
    suffix = f"_{kind}"
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
            label, path = label.strip(), path.strip()
        else:
            path = item.strip()
            name = Path(path).name
            if name.endswith(".csv.gz"):
                stem = name[: -len(".csv.gz")]
            elif name.endswith(".csv"):
                stem = name[: -len(".csv")]
            else:
                stem = Path(path).stem
            if stem.startswith("cmp_"):
                stem = stem[len("cmp_") :]
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
            label = stem
        if not label:
            raise ValueError(f"Invalid labeled path '{item}'")
        out[label] = path
    return out


def _load_cmp_topo(path, marker, default_length=None):
    df = pd.read_csv(path)
    required = {"id", "norm_rf", "weighted_rf", "kf_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["marker"] = marker
    if "n_tips" not in df.columns:
        df["n_tips"] = df["id"].map(_extract_n_tips)
    if "length" not in df.columns and default_length is not None:
        df["length"] = df["id"].map(lambda x: _extract_length(x, default_length))
    return df


def _load_cmp_dist_tree(path, marker, chunksize=2_000_000, id_to_n_tips=None):
    sums = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        required = {"id", "ref_dist", "cmp_dist"}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if "n_tips" not in chunk.columns:
            if id_to_n_tips is not None:
                chunk["n_tips"] = chunk["id"].map(id_to_n_tips)
            missing_n_tips = chunk["n_tips"].isna() if "n_tips" in chunk.columns else None
            if "n_tips" not in chunk.columns or missing_n_tips.any():
                unresolved = chunk["n_tips"].isna() if "n_tips" in chunk.columns else None
                if "n_tips" not in chunk.columns:
                    chunk["n_tips"] = chunk["id"].map(_extract_n_tips)
                else:
                    chunk.loc[unresolved, "n_tips"] = chunk.loc[unresolved, "id"].map(_extract_n_tips)
        chunk["MAE"] = (chunk["ref_dist"] - chunk["cmp_dist"]).abs()
        chunk["MRE"] = np.where(
            chunk["ref_dist"] > 0, chunk["MAE"] / chunk["ref_dist"], np.nan
        )
        # Aggregate chunk-level sums/counts so we never keep all pairwise rows in RAM.
        grouped = chunk.groupby(["id", "n_tips"], as_index=False).agg(
            MAE_sum=("MAE", "sum"),
            MAE_count=("MAE", "count"),
            MRE_sum=("MRE", "sum"),
            MRE_count=("MRE", "count"),
        )
        sums.append(grouped)
    if not sums:
        return pd.DataFrame(columns=["marker", "id", "n_tips", "MAE", "MRE"])
    merged = pd.concat(sums, ignore_index=True).groupby(
        ["id", "n_tips"], as_index=False
    ).sum()
    merged["MAE"] = merged["MAE_sum"] / merged["MAE_count"]
    merged["MRE"] = merged["MRE_sum"] / merged["MRE_count"]
    merged["marker"] = marker
    return merged[["marker", "id", "n_tips", "MAE", "MRE"]]


def _load_cmp_dist_pairwise(
    path,
    marker,
    sample_frac=1.0,
    chunksize=2_000_000,
    default_length=None,
    random_state=42,
):
    if not (0 < float(sample_frac) <= 1.0):
        raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}")

    kept = []
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
        required = {"id", "ref_dist", "cmp_dist"}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        if sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=random_state + i)
            if chunk.empty:
                continue

        chunk = chunk.copy()
        if "n_tips" not in chunk.columns:
            chunk["n_tips"] = chunk["id"].map(_extract_n_tips)
        if "length" not in chunk.columns and default_length is not None:
            chunk["length"] = chunk["id"].map(lambda x: _extract_length(x, default_length))

        chunk["marker"] = marker
        chunk["MAE"] = (chunk["ref_dist"] - chunk["cmp_dist"]).abs()
        chunk["MRE"] = np.where(
            chunk["ref_dist"] > 0, chunk["MAE"] / chunk["ref_dist"], np.nan
        )
        chunk["MRD"] = np.where(
            chunk["ref_dist"] > 0,
            (chunk["ref_dist"] - chunk["cmp_dist"]) / chunk["ref_dist"],
            np.nan,
        )
        kept.append(chunk)

    if not kept:
        cols = [
            "id",
            "ref_dist",
            "cmp_dist",
            "marker",
            "n_tips",
            "length",
            "MAE",
            "MRE",
            "MRD",
        ]
        return pd.DataFrame(columns=cols)
    return pd.concat(kept, ignore_index=True)


def _load_external_results(topo_path, dist_path, label, default_length=500):
    _register_external_style(label)
    topo = _load_cmp_topo(topo_path, label, default_length=default_length)
    dists = _load_cmp_dist_pairwise(
        dist_path,
        label,
        sample_frac=SAMPLING_FRAC,
        chunksize=1_000_000,
        default_length=default_length,
    )
    dists["dataset"] = "LG+GC"
    return topo, dists


def _load_lggc_topo_500(path):
    topo = pd.read_csv(path)
    topo = topo.copy()
    topo["length"] = topo["id"].apply(lambda x: int(str(x).split("_")[-1]))
    topo["marker"] = topo["marker"].apply(lambda x: RENAMER.get(x, x))
    if "n_tips" not in topo.columns:
        topo["n_tips"] = topo["id"].map(_extract_n_tips)
    return topo[topo["length"] == 500]


def _load_lggc_dist_tree_500(path, chunksize=2_000_000):
    # Stream the very large dists_lggc.csv and keep only PF methods needed by base_vs_ft.
    wanted = {"PF+FastME", "PF_Base+FastME"}
    sums = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk = chunk.copy()
        chunk["length"] = chunk["id"].apply(lambda x: int(str(x).split("_")[-1]))
        chunk = chunk[chunk["length"] == 500]
        if chunk.empty:
            continue
        chunk["marker"] = chunk["marker"].apply(lambda x: RENAMER.get(x, x))
        chunk = chunk[chunk["marker"].isin(wanted)]
        if chunk.empty:
            continue
        if "n_tips" not in chunk.columns:
            chunk["n_tips"] = chunk["id"].map(_extract_n_tips)
        chunk["MAE"] = (chunk["ref_dist"] - chunk["cmp_dist"]).abs()
        chunk["MRE"] = np.where(
            chunk["ref_dist"] > 0, chunk["MAE"] / chunk["ref_dist"], np.nan
        )
        grouped = chunk.groupby(["marker", "id", "n_tips"], as_index=False).agg(
            MAE_sum=("MAE", "sum"),
            MAE_count=("MAE", "count"),
            MRE_sum=("MRE", "sum"),
            MRE_count=("MRE", "count"),
        )
        sums.append(grouped)
    if not sums:
        return pd.DataFrame(columns=["marker", "id", "n_tips", "MAE", "MRE"])
    merged = pd.concat(sums, ignore_index=True).groupby(
        ["marker", "id", "n_tips"], as_index=False
    ).sum()
    merged["MAE"] = merged["MAE_sum"] / merged["MAE_count"]
    merged["MRE"] = merged["MRE_sum"] / merged["MRE_count"]
    return merged[["marker", "id", "n_tips", "MAE", "MRE"]]


def generate_base_vs_mre_plus_new_model(
    output_pdf="./figures/pfbase_quartet_close/base_vs_mre_plus_qclose.pdf",
    lggc_topo_path="./data/topos_lggc.csv",
    lggc_dist_path="./data/dists_lggc.csv",
    new_topo_path="./runs/pfbase_quartet_close/eval_val_lggc/cmp_quartet_close_topo.csv",
    new_dist_path="./runs/pfbase_quartet_close/eval_val_lggc/cmp_quartet_close_dist.csv",
    dist_chunksize=2_000_000,
    method_label="PF_QCLOSE+FastME",
):
    sns.set_context("notebook")
    sns.set_style("darkgrid")

    if not Path(new_topo_path).exists():
        raise FileNotFoundError(f"Missing new-model topo file: {new_topo_path}")
    if not Path(new_dist_path).exists():
        raise FileNotFoundError(f"Missing new-model dist file: {new_dist_path}")

    topo_lggc = _load_lggc_topo_500(lggc_topo_path)
    dist_lggc = _load_lggc_dist_tree_500(lggc_dist_path, chunksize=dist_chunksize)

    _register_style_alias(method_label)
    new_topo = _load_cmp_topo(new_topo_path, method_label)
    new_dist = _load_cmp_dist_tree(
        new_dist_path, method_label, chunksize=dist_chunksize
    )

    topo_plus = pd.concat([topo_lggc, new_topo], ignore_index=True)
    dist_plus = pd.concat([dist_lggc, new_dist], ignore_index=True)

    fig = base_vs_ft(
        topo_plus,
        dist_plus,
        (9, 8),
        methods=["PF+FastME", "PF_Base+FastME", method_label],
    )

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Saved: {output_pdf}")


def _metric_panel(ax, df, metric, ylabel):
    tips = sorted(df["n_tips"].dropna().unique().tolist())
    if len(tips) <= 1:
        means = df.groupby("marker")[metric].mean().reset_index()
        sns.barplot(data=means, x="marker", y=metric, ax=ax)
        ax.set_xlabel("Method")
        ax.tick_params(axis="x", rotation=20)
    else:
        for marker, sub in df.groupby("marker"):
            means = sub.groupby("n_tips", as_index=False)[metric].mean()
            ax.plot(means["n_tips"], means[metric], marker="o", label=marker)
        ax.set_xlabel("Number of leaves")
        ax.legend(loc="best", fontsize="small")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)


def run_cmp_mode(cmp_topo_items, cmp_dist_items, outdir, dist_chunksize):
    if not cmp_topo_items or not cmp_dist_items:
        raise ValueError("Both --cmp-topo and --cmp-dist are required in cmp mode.")

    topo_paths = _parse_labelled_paths(cmp_topo_items, "topo")
    dist_paths = _parse_labelled_paths(cmp_dist_items, "dist")
    common = sorted(set(topo_paths).intersection(dist_paths))
    if not common:
        raise ValueError(
            "No shared labels between --cmp-topo and --cmp-dist. "
            "Use LABEL=path for both sides (e.g. PF=... PF_QCLOSE=...)."
        )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    topo_df = pd.concat(
        [_load_cmp_topo(topo_paths[label], label) for label in common],
        ignore_index=True,
    )
    topo_maps = {
        label: (
            topo_df[topo_df["marker"] == label]
            .drop_duplicates(subset=["id"])
            .set_index("id")["n_tips"]
            .to_dict()
        )
        for label in common
    }
    dist_tree = pd.concat(
        [
            _load_cmp_dist_tree(
                dist_paths[label],
                label,
                chunksize=max(int(dist_chunksize), 1),
                id_to_n_tips=topo_maps.get(label),
            )
            for label in common
        ],
        ignore_index=True,
    )

    topo_tree = topo_df.groupby(["marker", "id", "n_tips"], as_index=False)[
        ["norm_rf", "weighted_rf", "kf_score"]
    ].mean()

    topo_summary = topo_tree.groupby("marker").agg(
        n_trees=("id", "nunique"),
        mean_norm_rf=("norm_rf", "mean"),
        median_norm_rf=("norm_rf", "median"),
        mean_weighted_rf=("weighted_rf", "mean"),
        median_weighted_rf=("weighted_rf", "median"),
        mean_kf_score=("kf_score", "mean"),
        median_kf_score=("kf_score", "median"),
    )
    dist_summary = dist_tree.groupby("marker").agg(
        n_trees_dist=("id", "nunique"),
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        mean_MRE=("MRE", "mean"),
        median_MRE=("MRE", "median"),
    )
    summary = topo_summary.join(dist_summary, how="outer").reset_index()
    summary.to_csv(outdir / "summary_metrics.csv", index=False)
    topo_tree.to_csv(outdir / "topo_tree_metrics.csv", index=False)
    dist_tree.to_csv(outdir / "dist_tree_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    _metric_panel(
        axes[0, 0], topo_tree, "norm_rf", "Normalized Robinson-Foulds distance"
    )
    _metric_panel(
        axes[0, 1], topo_tree, "weighted_rf", "Weighted Robinson-Foulds distance"
    )
    _metric_panel(axes[0, 2], topo_tree, "kf_score", "Kuhner-Felsenstein distance")
    _metric_panel(axes[1, 0], dist_tree, "MAE", "Mean Absolute Error")
    _metric_panel(axes[1, 1], dist_tree, "MRE", "Mean Relative Error")
    axes[1, 2].axis("off")
    fig.savefig(outdir / "cmp_summary_metrics.png", dpi=160)
    plt.close(fig)

    print(f"[cmp-mode] Saved outputs to: {outdir}")
    print(f"[cmp-mode] Summary table: {outdir / 'summary_metrics.csv'}")


def label_axes(axes, fig, uppercase=False):
    """
    Add letter labels in the upper left corner of each subplot
    """

    letters = "abcdefghijklmnopqrstuvwxyz"
    if uppercase:
        letters = letters.upper()
    labels = [f"{l})" for l in letters]

    # Set axis labels
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ctx = sns.plotting_context()
    sty = sns.axes_style()
    for ax, label in zip(axes, labels):
        ax.text(
            0.0,
            1.0,
            label,
            transform=ax.transAxes + trans,
            fontsize=ctx["axes.titlesize"],
            verticalalignment="top",
            fontfamily=sty["font.family"][0],
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3.0),
        )


def group_elapsed(path, rename_pf_cpu: bool = False):
    # Reading and parsing execution metadata
    exec = pd.read_csv(path)

    if rename_pf_cpu:
        exec["marker"] = exec.apply(
            lambda row: (
                "PF_CPU+FastME_outer"
                if row["marker"] == "PF_CPU+FastME" and row["timer"] == "times_outer"
                else row["marker"]
            ),
            axis=1,
        )

    # Sum times and memory for 2 part methods (e.g. PF or Hamming)
    grouped = (
        exec[exec["timer"] != "times_pf_old"]  # Temp
        .groupby(["marker", "id"])[["elapsed_sec", "MaxRSS_kb"]]
        .sum()
        .reset_index()
    )
    grouped["n_tips"] = grouped["id"].apply(lambda x: x.split("_")[1]).astype(int)
    grouped["length"] = grouped["id"].apply(lambda x: x.split("_")[-1]).astype(int)

    return grouped


def plot_line(
    df,
    method,
    ax,
    xvar="n_tips",
    yvar="norm_rf",
    label=True,
    alpha=1.0,
    offset=0.0,
):
    """
    Call sns.lineplot on a particular subset of df and show in on a given Axes object
    """
    col, ls, marker = STYLES[method]
    sub = df[df["marker"] == method]
    sns.lineplot(
        x=sub[xvar],
        y=sub[yvar] + offset,
        color=col,
        ls=ls,
        label=method if label else None,
        alpha=alpha,
        ax=ax,
        marker=marker,
    )


def build_plot(df, methods, metric, label, figsize, xticks, ymajor, yminor):
    """Build the 3 subplots for different alignment lengths on LG+GC"""

    # Setup figure and layout
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(3, 4, figure=fig, top=0.95, bottom=0.05, right=0.5, left=0.05)
    ax1 = fig.add_subplot(gs[:2, :])
    ax2 = fig.add_subplot(gs[-1, :2], sharey=ax1)
    ax3 = fig.add_subplot(gs[-1, 2:], sharey=ax1)

    for length, ax in zip([500, 250, 1000], [ax1, ax2, ax3]):
        # Plot data to subplot
        for method in methods:
            plot_line(df[df["length"] == length], method, ax, "n_tips", metric)
        ax.set_title(f"Alignment length = {length}")
        ax.set_xlabel("Number of leaves")
        # Supress ylabel of lower right subplot
        if length == 1000:
            ax.set_ylabel("")
        else:
            ax.set_ylabel(label)

    plt.setp(ax3.get_yticklabels(), visible=False)

    # Custom ticks for top subplot
    if xticks is not None:
        ax1.set_xticks(xticks)
    if ymajor is not None:
        ax1.set_yticks(ymajor)
    if yminor is not None:
        ax1.set_yticks(yminor, minor=True)
    ax1.minorticks_on()
    ax1.grid(which="major", ls="-", linewidth=1.5, color="white")
    ax1.grid(which="minor", ls="-", linewidth=0.5, color="white", axis="y")

    # Add labels to sunplots
    label_axes([ax1, ax2, ax3], fig)

    # Add legends
    ax1.legend(
        loc="upper left",
        title="Tree inference\nmethod",
        bbox_to_anchor=(1, 1),
    )
    ax2.get_legend().remove()
    ax3.get_legend().remove()

    return fig


def side_by_side(df, methods, metric, label, figsize):
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(4, 9, figure=fig, top=0.95, bottom=0.05, right=0.5, left=0.05)
    ax1 = fig.add_subplot(gs[:3, :3])
    ax2 = fig.add_subplot(gs[:3, 3:6], sharey=ax1)
    ax3 = fig.add_subplot(gs[:3, -3:], sharey=ax1)
    ax_legend = fig.add_subplot(gs[-1, :])

    axes = [ax1, ax2, ax3]

    for length, ax in zip([250, 500, 1000], axes):
        # Plot data to subplot
        for method in methods:
            plot_line(df[df["length"] == length], method, ax, "n_tips", metric)
        ax.set_title(f"Alignment length = {length}")
        ax.set_xlabel("Number of leaves")

        # Set Y axis labels
        if length == 250:
            ax.set_ylabel(label)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylabel("")

    h, l = ax1.get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()

    ax_legend.set_axis_off()
    ax_legend.set_ylabel("")
    ax_legend.set_xlabel("")
    ax_legend.legend(h, l, loc="center", bbox_to_anchor=(0.5, 0.5), ncol=4)

    return fig


def build_LGGC_normRF(df, figsize):
    label = "Normalized Robinson-Foulds distance"
    return side_by_side(df, LGGC_METHODS_NO_HAMMING, "norm_rf", label, figsize)


def build_LGGC_KFscore(df, figsize):
    label = "Kuhner-Felsenstein distance"
    return side_by_side(df, LGGC_METHODS_NO_HAMMING, "kf_score", label, figsize)


def build_LGGC_wRF(df, figsize):
    label = "weighted Robinson-Foulds distance"
    return side_by_side(df, LGGC_METHODS_NO_HAMMING, "weighted_rf", label, figsize)


def build_LGGC_lik(df, figsize):
    label = "Log-likelihood Ratio"
    fig = side_by_side(df, LGGC_METHODS_NO_HAMMING, "ratio", label, figsize)

    for ax in fig.axes[:-1]:
        ax.axhline(y=1, ls=":", color="gray")

    return fig


def single_LGGC_normRF(df, figsize, methods=None):
    if methods is None:
        methods = LGGC_METHODS_NO_HAMMING
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in methods:
        plot_line(sub, method, ax, "n_tips", "norm_rf")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Normalized Robinson-Foulds distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=min(len(methods), 4),
    )

    return fig


def single_LGGC_KFscore(df, figsize, methods=None):
    if methods is None:
        methods = LGGC_METHODS_NO_HAMMING
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in methods:
        plot_line(sub, method, ax, "n_tips", "kf_score")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Kuhner-Felsenstein distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=min(len(methods), 4),
    )

    return fig


def single_LGGC_wRF(df, figsize, methods=None):
    if methods is None:
        methods = LGGC_METHODS_NO_HAMMING
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in methods:
        plot_line(sub, method, ax, "n_tips", "weighted_rf")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("weighted Robinson-Foulds distance")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=min(len(methods), 4),
    )

    return fig


def single_LGGC_mae(df, figsize, methods=None):
    if methods is None:
        methods = LGGC_METHODS_NO_HAMMING
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in methods:
        plot_line(sub, method, ax, "n_tips", "MAE")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Mean Absolute Error on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=min(len(methods), 4),
    )

    return fig


def single_LGGC_mre(df, figsize, methods=None):
    if methods is None:
        methods = LGGC_METHODS_NO_HAMMING
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in methods:
        plot_line(sub, method, ax, "n_tips", "MRE")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Mean Relative Error on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(len(methods), 4),
    )

    return fig


def single_LGGC_mrd(df, figsize):
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in LGGC_METHODS_NO_HAMMING:
        plot_line(sub, method, ax, "n_tips", "MRD")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Mean Relative Difference on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )

    return fig


def single_LGGC_quantiles_mae(sub, figsize):
    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["PF_Base+FastME"]):
        plot_line(sub, method, ax, "percentile", "MAE")
    ax.set_xlabel("Reference Pairwise Distance Percentile")
    ax.set_ylabel("Mean Absolute Error on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.set_yscale("log")
    ax.set_xscale("log")

    return fig


def single_LGGC_quantiles_mre(sub, figsize):
    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["PF_Base+FastME"]):
        plot_line(sub, method, ax, "percentile", "MRE")
    ax.set_xlabel("Reference Pairwise Distance Percentile")
    ax.set_ylabel("Mean Relative Error on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.set_xscale("log")

    return fig


def single_LGGC_quantiles_mrd(sub, figsize):
    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["PF_Base+FastME"]):
        plot_line(sub, method, ax, "percentile", "MRD")
    ax.set_xlabel("Reference Pairwise Distance Percentile")
    ax.set_ylabel("Mean Relative Difference on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.set_xscale("log")

    return fig


def single_LGGC_binned_mae(sub, figsize):
    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["PF_Base+FastME"]):
        plot_line(sub, method, ax, "binned", "MAE")
    ax.set_xlabel("Binned Reference Pairwise Distance")
    ax.set_ylabel("Mean Absolute Error on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.set_yscale("log")
    ax.set_xscale("log")

    return fig


def single_LGGC_binned_mre(sub, figsize):
    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["PF_Base+FastME"]):
        plot_line(sub, method, ax, "binned", "MRE")
    ax.set_xlabel("Binned Reference Pairwise Distance Percentile")
    ax.set_ylabel("Mean Relative Error on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.set_xscale("log")

    return fig


def single_LGGC_binned_mrd(sub, figsize):
    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["PF_Base+FastME"]):
        plot_line(sub, method, ax, "binned", "MRD")
    ax.set_xlabel("Binned Reference Pairwise Distance Percentile")
    ax.set_ylabel("Mean Relative Difference on pairwise distance")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.set_xscale("log")

    return fig


def single_LGGC_elapsed(df, figsize, model_load_time=None):
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    if model_load_time is not None:
        plot_line(
            sub,
            "PF+FastME",
            ax,
            "n_tips",
            "elapsed_sec",
            offset=model_load_time,
            alpha=0.2,
            label=False,
        )
    for method in LGGC_METHODS_NO_HAMMING:
        plot_line(sub, method, ax, "n_tips", "elapsed_sec")

    ax.set_yscale("log")
    ax.grid(
        which="minor",
        ls=sns.axes_style()["grid.linestyle"],
        linewidth=0.4 * sns.plotting_context()["grid.linewidth"],
        color=sns.axes_style()["grid.color"],
        axis="y",
    )

    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Elapsed time (sec)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    return fig


def paper_elapsed(df, figsize, model_load_time=None):
    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    if model_load_time is not None:
        plot_line(
            df,
            "PF+FastME",
            ax,
            "n_tips",
            "elapsed_sec",
            offset=model_load_time,
            alpha=0.2,
            label=False,
        )    
    for method in sorted(LGGC_METHODS_NO_HAMMING + ["IQTree_MF"]):
        plot_line(df, method, ax, "n_tips", "elapsed_sec")

    ax.set_yscale("log")
    ax.grid(
        which="minor",
        ls=sns.axes_style()["grid.linestyle"],
        linewidth=0.4 * sns.plotting_context()["grid.linewidth"],
        color=sns.axes_style()["grid.color"],
        axis="y",
    )

    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Elapsed time (sec)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    return fig


def single_LGGC_mem(df, figsize):
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in LGGC_METHODS_NO_HAMMING:
        plot_line(sub, method, ax, "n_tips", "MaxRSS_kb")

    ax.set_yscale("log")
    ax.grid(
        which="minor",
        ls=sns.axes_style()["grid.linestyle"],
        linewidth=0.4 * sns.plotting_context()["grid.linewidth"],
        color=sns.axes_style()["grid.color"],
        axis="y",
    )

    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Maximum RSS (kB)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    return fig


def single_LGGC_lik(df, figsize):
    sub = df[df["length"] == 500]

    # Single RF plot for LG+GC 500 AAs
    fig, ax = plt.subplots(1, figsize=figsize, layout="constrained")
    for method in LGGC_METHODS_NO_HAMMING:
        plot_line(sub, method, ax, "n_tips", "ratio")
    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("Log-likelihood Ratio")
    ax.axhline(y=1, ls=":", color="gray", label="True Tree")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    return fig


def cherry_pastek_plots(
    cherry_df,
    pastek_df,
    figsize,
    yvar,
    ylabel,
    log=False,
    include_PF=True,
    sharey=False,
):
    handles = []
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(7, 6, figure=fig)
    kw = dict()
    ax_cherry = fig.add_subplot(gs[:6, :3])
    if sharey:
        kw["sharey"] = ax_cherry
    ax_pastek = fig.add_subplot(gs[:6, -3:], **kw)
    ax_dummy = fig.add_subplot(gs[-1, :])

    ms = sns.plotting_context()["lines.markersize"]
    marker_kwargs = dict(
        markersize=ms,
        markeredgecolor="w",
        markeredgewidth=0.75,
    )

    methods = FINE_TUNE_METHODS if include_PF else FINE_TUNE_METHODS[:-1]

    # Cherry plot
    for method in methods + ["PF_Cherry+FastME"]:
        plot_line(cherry_df, method, ax_cherry, "n_tips", yvar)
        col, ls, marker = STYLES[method]
        col, ls, marker = STYLES[method]
        handles.append(
            Line2D(
                [0], [0], color=col, ls=ls, label=method, marker=marker, **marker_kwargs
            )
        )

    ax_cherry.set_title("Cherry")

    # Pastek plot
    for method in methods + ["PF_SelReg+FastME"]:
        plot_line(pastek_df, method, ax_pastek, "n_tips", yvar)
    col, ls, marker = STYLES["PF_SelReg+FastME"]
    handles.append(
        Line2D(
            [0],
            [0],
            color=col,
            ls=ls,
            label="PF_SelReg+FastME",
            marker=marker,
            **marker_kwargs,
        )
    )
    ax_pastek.set_title("SelReg")

    # Set log axis if needed
    if log:
        for ax in [ax_cherry, ax_pastek]:
            ax.set_yscale("log")
            ax.grid(
                which="minor",
                ls=sns.axes_style()["grid.linestyle"],
                linewidth=0.4 * sns.plotting_context()["grid.linewidth"],
                color=sns.axes_style()["grid.color"],
                axis="y",
            )

    # Add subplot label
    label_axes([ax_cherry, ax_pastek], fig)

    # Set axis labels
    ax_cherry.set_ylabel(ylabel)
    for ax in [ax_cherry, ax_pastek]:
        ax.set_xlabel("Number of leaves")
        ax.get_legend().remove()

    ax_pastek.set_ylabel("")
    if sharey:
        plt.setp(ax_pastek.get_yticklabels(), visible=False)

    ax_dummy.set_axis_off()
    ax_dummy.set_ylabel("")
    ax_dummy.set_xlabel("")
    ax_dummy.legend(handles=handles, loc="center", ncol=3)

    return fig


def fine_tuned_plot(
    gaps_df,
    cherry_df,
    pastek_df,
    figsize,
    yvar,
    ylabel,
    log=False,
    include_PF=True,
    exclude_pf_gaps=True,
    sharey=False,
):
    handles = []
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(7, 6, figure=fig)
    ax_gaps = fig.add_subplot(gs[:6, :2])
    kw = dict()
    if sharey:
        kw["sharey"] = ax_gaps
    ax_cherry = fig.add_subplot(gs[:6, 2:-2], **kw)
    ax_pastek = fig.add_subplot(gs[:6, -2:], **kw)
    ax_dummy = fig.add_subplot(gs[-1, :])

    ms = sns.plotting_context()["lines.markersize"]
    marker_kwargs = dict(
        markersize=ms,
        markeredgecolor="w",
        markeredgewidth=0.75,
    )

    methods = FINE_TUNE_METHODS if include_PF else FINE_TUNE_METHODS[:-1]

    # Gaps plot
    for method in methods + ["PF_Indel+FastME"]:
        if exclude_pf_gaps and method == "PF+FastME":
            continue
        plot_line(gaps_df, method, ax_gaps, "n_tips", yvar)
        col, ls, marker = STYLES[method]
        handles.append(
            Line2D(
                [0], [0], color=col, ls=ls, label=method, marker=marker, **marker_kwargs
            )
        )
    ax_gaps.set_title("LG+GC+Gaps")

    # Cherry plot
    for method in methods + ["PF_Cherry+FastME"]:
        plot_line(cherry_df, method, ax_cherry, "n_tips", yvar)
        col, ls, marker = STYLES[method]
    col, ls, marker = STYLES["PF_Cherry+FastME"]
    handles.append(
        Line2D(
            [0],
            [0],
            color=col,
            ls=ls,
            label="PF_SelReg+FastME",
            marker=marker,
            **marker_kwargs,
        )
    )
    ax_cherry.set_title("Cherry")

    # Pastek plot
    for method in methods + ["PF_SelReg+FastME"]:
        plot_line(pastek_df, method, ax_pastek, "n_tips", yvar)
    col, ls, marker = STYLES["PF_SelReg+FastME"]
    handles.append(
        Line2D(
            [0],
            [0],
            color=col,
            ls=ls,
            label="PF_SelReg+FastME",
            marker=marker,
            **marker_kwargs,
        )
    )
    ax_pastek.set_title("SelReg")

    # Set log axis if needed
    if log:
        for ax in [ax_gaps, ax_cherry, ax_pastek]:
            ax.set_yscale("log")
            ax.grid(
                which="minor",
                ls=sns.axes_style()["grid.linestyle"],
                linewidth=0.4 * sns.plotting_context()["grid.linewidth"],
                color=sns.axes_style()["grid.color"],
                axis="y",
            )

    # Add subplot labels
    label_axes([ax_gaps, ax_cherry, ax_pastek], fig)

    # Set axis labels
    ax_gaps.set_ylabel(ylabel)
    for ax in [ax_gaps, ax_cherry, ax_pastek]:
        ax.set_xlabel("Number of leaves")
        ax.get_legend().remove()

    ax_cherry.set_ylabel("")
    ax_pastek.set_ylabel("")
    if sharey:
        plt.setp(ax_cherry.get_yticklabels(), visible=False)
        plt.setp(ax_pastek.get_yticklabels(), visible=False)

    ax_dummy.set_axis_off()
    ax_dummy.set_ylabel("")
    ax_dummy.set_xlabel("")
    ax_dummy.legend(handles=handles, loc="center", ncol=3)

    return fig


def fine_tuned_normRF(gaps_df, cherry_df, pastek_df, figsize):
    return fine_tuned_plot(
        gaps_df,
        cherry_df,
        pastek_df,
        figsize,
        "norm_rf",
        "Normalized Robinson-Foulds distance",
    )


def cherry_pastek_normRF(cherry_df, pastek_df, figsize):
    return cherry_pastek_plots(
        cherry_df,
        pastek_df,
        figsize,
        "norm_rf",
        "Normalized Robinson-Foulds distance",
        sharey=True,
    )


def fine_tuned_KFscore(gaps_df, cherry_df, pastek_df, figsize):
    return fine_tuned_plot(
        gaps_df,
        cherry_df,
        pastek_df,
        figsize,
        "kf_score",
        "Kuhner-Felsenstein distance",
    )


def cherry_pastek_KFscore(cherry_df, pastek_df, figsize):
    return cherry_pastek_plots(
        cherry_df,
        pastek_df,
        figsize,
        "kf_score",
        "Kuhner-Felsenstein distance",
        sharey=True,
    )


def fine_tuned_wRF(gaps_df, cherry_df, pastek_df, figsize):
    return fine_tuned_plot(
        gaps_df,
        cherry_df,
        pastek_df,
        figsize,
        "weighted_rf",
        "weighted Robinson-Foulds distance",
    )


def cherry_pastek_wRF(cherry_df, pastek_df, figsize):
    return cherry_pastek_plots(
        cherry_df,
        pastek_df,
        figsize,
        "weighted_rf",
        "weighted Robinson-Foulds distance",
        sharey=True,
    )


def fine_tuned_elapsed(gaps_df, cherry_df, pastek_df, figsize):
    return fine_tuned_plot(
        gaps_df,
        cherry_df,
        pastek_df,
        figsize,
        "elapsed_sec",
        "Elapsed time (sec)",
        log=True,
        include_PF=False,
        sharey=True,
    )


def fine_tuned_mem(gaps_df, cherry_df, pastek_df, figsize):
    return fine_tuned_plot(
        gaps_df,
        cherry_df,
        pastek_df,
        figsize,
        "MaxRSS_kb",
        "Maximum RSS (kB)",
        log=True,
        include_PF=False,
        sharey=True,
    )


def fine_tuned_mae(gaps_df, cherry_df, pastek_df, figsize):
    return fine_tuned_plot(gaps_df, cherry_df, pastek_df, figsize, "MAE", "MAE")


def cherry_pastek_topos(
    cherry_df,
    pastek_df,
    figsize,
    include_PF=False,
):
    # Setup figure and layout
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(7, 6, figure=fig, top=0.95, bottom=0.05, right=0.5, left=0.05)

    rfc_ax = fig.add_subplot(gs[:3, :3])
    rfp_ax = fig.add_subplot(gs[:3, -3:], sharey=rfc_ax)
    kfc_ax = fig.add_subplot(gs[3:-1, :3])
    kfp_ax = fig.add_subplot(gs[3:-1, -3:], sharey=kfc_ax)
    legend_ax = fig.add_subplot(gs[-1, :])

    handles = []

    ms = sns.plotting_context()["lines.markersize"]
    marker_kwargs = dict(
        markersize=ms,
        markeredgecolor="w",
        markeredgewidth=0.75,
    )

    methods = FINE_TUNE_METHODS if include_PF else FINE_TUNE_METHODS[:-1]

    # Cherry plot
    for method in methods + ["PF_Cherry+FastME"]:
        plot_line(cherry_df, method, rfc_ax, "n_tips", "norm_rf")
        plot_line(cherry_df, method, kfc_ax, "n_tips", "kf_score")
        col, ls, marker = STYLES[method]
        col, ls, marker = STYLES[method]
        handles.append(
            Line2D(
                [0], [0], color=col, ls=ls, label=method, marker=marker, **marker_kwargs
            )
        )

    rfc_ax.set_title("Cherry")

    # Pastek plot
    for method in methods + ["PF_SelReg+FastME"]:
        plot_line(pastek_df, method, rfp_ax, "n_tips", "norm_rf")
        plot_line(pastek_df, method, kfp_ax, "n_tips", "kf_score")

    col, ls, marker = STYLES["PF_SelReg+FastME"]
    handles.append(
        Line2D(
            [0],
            [0],
            color=col,
            ls=ls,
            label="PF_SelReg+FastME",
            marker=marker,
            **marker_kwargs,
        )
    )
    rfp_ax.set_title("SelReg")

    # Add subplot label
    label_axes([rfc_ax, rfp_ax, kfc_ax, kfp_ax], fig)

    # Remove legends
    for ax in [kfc_ax, kfp_ax, rfc_ax, rfp_ax]:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_legend().remove()

    # Set X axis labels
    for ax in [kfc_ax, kfp_ax]:
        ax.set_title("")
        ax.set_xlabel("Number of leaves")

    # Set Y axis labels
    rfc_ax.set_ylabel("Normalized Robinson-Foulds distance")
    kfc_ax.set_ylabel("Kuhner-Felsenstein distance")
    for ax in [rfp_ax, kfp_ax]:
        ax.set_ylabel("")
        plt.setp(ax.get_yticklabels(), visible=False)

    legend_ax.set_axis_off()
    legend_ax.set_ylabel("")
    legend_ax.set_xlabel("")
    legend_ax.legend(handles=handles, loc="center", ncol=3)
    return fig


def base_vs_ft(
    topo_df,
    dists_df,
    figsize,
    methods=None,
):
    # Setup figure and layout
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(7, 6, figure=fig, top=0.95, bottom=0.05, right=0.5, left=0.05)

    kf_ax = fig.add_subplot(gs[:3, :3])
    rf_ax = fig.add_subplot(gs[:3, -3:])
    mae_ax = fig.add_subplot(gs[3:-1, :3])
    mre_ax = fig.add_subplot(gs[3:-1, -3:])
    legend_ax = fig.add_subplot(gs[-1, :])

    if methods is None:
        methods = ["PF+FastME", "PF_Base+FastME"]
    kws = dict(x="n_tips", hue="marker", hue_order=methods, marker="o")

    # Plot figures
    sns.lineplot(data=topo_df, y="kf_score", ax=kf_ax, **kws)
    sns.lineplot(data=topo_df, y="norm_rf", ax=rf_ax, **kws)
    sns.lineplot(data=dists_df, y="MAE", ax=mae_ax, **kws)
    sns.lineplot(data=dists_df, y="MRE", ax=mre_ax, **kws)

    handles, labels = mre_ax.get_legend_handles_labels()

    # Add subplot label
    label_axes([kf_ax, rf_ax, mae_ax, mre_ax], fig)

    # Remove legends
    for ax in [mae_ax, mre_ax, kf_ax, rf_ax]:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_legend().remove()

    # Set X axis labels
    for ax in [mae_ax, mre_ax]:
        ax.set_title("")
        ax.set_xlabel("Number of leaves")

    # Set Y axis labels
    rf_ax.set_ylabel("Normalized Robinson-Foulds distance")
    kf_ax.set_ylabel("Kuhner-Felsenstein distance")
    mae_ax.set_ylabel("Mean Absolute Error")
    mre_ax.set_ylabel("Mean Relative Error")

    for ax in [rf_ax, kf_ax]:
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), visible=False)

    legend_ax.set_axis_off()
    legend_ax.set_ylabel("")
    legend_ax.set_xlabel("")
    legend_ax.legend(handles, labels, loc="center", ncol=min(len(methods), 3))
    return fig


def dataset_plot(df, mae_df, figsize, methods):
    # Setup figure and layout
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(4, 5, figure=fig, top=0.95, bottom=0.05, right=0.5, left=0.05)

    kf_ax = fig.add_subplot(gs[:3, :3])
    mae_ax = fig.add_subplot(gs[:2, -2:])
    rf_ax = fig.add_subplot(gs[-2:, -2:], sharex=mae_ax)
    legend_ax = fig.add_subplot(gs[-1, :3])

    # Plot lines
    for method in methods:
        plot_line(df, method, kf_ax, yvar="kf_score")
        plot_line(df, method, rf_ax, yvar="norm_rf")
        plot_line(mae_df, method, mae_ax, yvar="MAE")

    # Add legend
    legend_ax.set_axis_off()
    legend_ax.set_ylabel("")
    legend_ax.set_xlabel("")
    handles, labels = kf_ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", ncol=3)

    # Set axis labels
    plot_axes = [kf_ax, mae_ax, rf_ax]
    label_axes(plot_axes, fig)
    for ax in plot_axes:
        ax.get_legend().remove()
        ax.set_xlabel("Number of leaves")

    kf_ax.set_ylabel("Kuhner-Felsenstein distance")
    mae_ax.set_ylabel("Mean absolute error")
    rf_ax.set_ylabel("Normalized Robinson-Foulds distance")

    return fig


def hist_LGGC(dists, figsize):
    return hist_4x4(dists, figsize, LGGC_METHODS_NO_HAMMING)


def hist_cherry_4x4(dists, figsize):
    return hist_4x4(
        dists, figsize, sorted(FINE_TUNE_METHODS[:-1] + ["PF_Cherry+FastME"])
    )


def hist_pastek_4x4(dists, figsize):
    return hist_4x4(
        dists, figsize, sorted(FINE_TUNE_METHODS[:-1] + ["PF_SelReg+FastME"])
    )


def hist_4x4(dists, figsize, methods):
    assert len(methods) == 4  # So we dont break the layout

    # Safety barrier
    s = dists[(dists["ref_dist"] > 0) & (dists["cmp_dist"] > 0)]

    # Compute n of bins and common color scale
    _, bin_edges = np.histogram(
        np.log10(s.loc[s["marker"].isin(methods), "ref_dist"]),
        bins="auto",
    )
    bin_nr = len(bin_edges) - 1
    vmin_list, vmax_list = [], []
    for c_type in methods:
        arr, _, _ = np.histogram2d(
            np.log10(s.loc[s.marker == c_type, "ref_dist"]),
            np.log10(s.loc[s.marker == c_type, "cmp_dist"]),
            bins=bin_nr,
        )
        # To get Frequencies instead of counts
        vmin_list.append(arr.min() / (s.marker == c_type).sum())
        vmax_list.append(arr.max() / (s.marker == c_type).sum())

    # find lowest and highest counts for all subplots
    vmin_all = min(vmin_list)
    vmax_all = max(vmax_list)

    fig = plt.figure(layout="constrained", figsize=figsize)
    base = 15
    gs = GridSpec(2 * base, 2 * base + 1, figure=fig)
    ax1 = fig.add_subplot(gs[:base, :base])
    ax2 = fig.add_subplot(gs[:base, base:-1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[-base:, :base], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[-base:, base:-1], sharex=ax1, sharey=ax1)
    cbax = fig.add_subplot(gs[:, -1:])

    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ctx = sns.plotting_context()
    sty = sns.axes_style()
    for method, ax in zip(methods, [ax1, ax2, ax3, ax4]):
        sns.histplot(
            data=s[s["marker"] == method],
            x="ref_dist",
            y="cmp_dist",
            log_scale=True,
            stat="proportion",
            bins=bin_nr,
            vmin=vmin_all,
            vmax=vmax_all,
            cbar=True,
            cbar_ax=cbax,
            ax=ax,
        )
        ax.text(
            0.0,
            1.0,
            method,
            transform=ax.transAxes + trans,
            fontsize=ctx["axes.titlesize"],
            verticalalignment="top",
            fontfamily=sty["font.family"][0],
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3.0),
        )

        # Label Y axes
        if ax in [ax1, ax3]:
            ax.set_ylabel("Predicted distance")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        # Lavel X axes
        if ax in [ax3, ax4]:
            ax.set_xlabel("Reference distance")
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # Plot reference line
        ax.axline(
            xy1=(1e-2, 1e-2),
            xy2=(1, 1),
            ls=":",
            color="red",
            zorder=0.5,
            lw=ctx["grid.linewidth"],
        )

    return fig


def hist_ft(dists, figsize, methods):
    # methods = FINE_TUNE_METHODS + ["PF_SelReg+FastME"]

    # Safety barrier
    s = dists[(dists["ref_dist"] > 0) & (dists["cmp_dist"] > 0)]

    # Compute n of bins and common color scale
    _, bin_edges = np.histogram(
        np.log10(s.loc[s["marker"].isin(methods), "ref_dist"]),
        bins="auto",
    )
    bin_nr = len(bin_edges) - 1
    vmin_list, vmax_list = [], []
    for c_type in methods:
        arr, _, _ = np.histogram2d(
            np.log10(s.loc[s.marker == c_type, "ref_dist"]),
            np.log10(s.loc[s.marker == c_type, "cmp_dist"]),
            bins=bin_nr,
        )
        vmin_list.append(arr.min())
        vmax_list.append(arr.max())

    # find lowest and highest counts for all subplots
    vmin_all = min(vmin_list)
    vmax_all = max(vmax_list)

    fig = plt.figure(layout="constrained", figsize=figsize)
    base = 15
    gs = GridSpec(3 * base, 2 * base + 1, figure=fig)
    iqax = fig.add_subplot(gs[:base, :base])
    ftax = fig.add_subplot(gs[:base, base:-1], sharex=iqax, sharey=iqax)
    fmax = fig.add_subplot(gs[base:-base, :base], sharex=iqax, sharey=iqax)
    pfax = fig.add_subplot(gs[base:-base, base:-1], sharex=iqax, sharey=iqax)
    ptax = fig.add_subplot(gs[-base:, :base], sharex=iqax, sharey=iqax)
    cbax = fig.add_subplot(gs[1:-1, -1:])

    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ctx = sns.plotting_context()
    sty = sns.axes_style()
    for method, ax in zip(methods, [iqax, ftax, fmax, pfax, ptax]):
        sns.histplot(
            data=s[s["marker"] == method],
            x="ref_dist",
            y="cmp_dist",
            log_scale=True,
            stat="count",
            bins=bin_nr,
            vmin=vmin_all,
            vmax=vmax_all,
            cbar=True,
            cbar_ax=cbax,
            ax=ax,
        )
        ax.text(
            0.0,
            1.0,
            method,
            transform=ax.transAxes + trans,
            fontsize=ctx["axes.titlesize"],
            verticalalignment="top",
            fontfamily=sty["font.family"][0],
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3.0),
        )
        ax.axline(
            xy1=(1e-2, 1e-2), xy2=(1, 1), ls=":", color="white", zorder=2, alpha=0.5
        )

    # Set axes
    for ax in [iqax, fmax, ptax]:
        ax.set_ylabel("Predicted distance")
    for ax in [ftax, pfax]:
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    for ax in [pfax, ptax]:
        ax.set_xlabel("Reference distance")
    for ax in [iqax, fmax, ftax]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    return fig


def misspecification(means, figsize, datasets, methods):
    ax_titles = dict(
        kf_score="Kuhner-Felsenstein distance",
        norm_rf="Normalized Robinson-Foulds distance",
        MAE="Mean Absolute Error",
        MRE="Mean Relative Error",
    )

    def get_heatmap(df, var):
        return df.reset_index().pivot_table(
            index="dataset", columns="marker", values=var
        )

    fig, axes = plt.subplots(
        2, 2, sharex=True, sharey=True, figsize=figsize, layout="constrained"
    )
    for ax, var in zip(axes.flatten(), ax_titles):
        sns.heatmap(
            get_heatmap(means, var).loc[datasets, methods],
            ax=ax,
            square=True,
            fmt=".1e",
            annot=True,
            annot_kws=dict(fontsize="small"),
        )
        # Set titles
        ax.set_title(ax_titles[var])
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    # Remove ticks for internal axes
    for ax in axes[0, :]:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for ax in axes[:, 1]:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Rotate ticks for external axes
    for ax in axes[1, :]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for ax in axes[:, 0]:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    return fig


def plot_brlen_dists(sub, figsize):
    tip_order = sorted(sub[legend_name].unique())
    pal = sns.cubehelix_palette(len(tip_order))

    # Distribution of branch lengths per tree size in testing set
    fig, axes = plt.subplots(
        ncols=2, layout="constrained", figsize=figsize, sharey=True
    )
    sns.kdeplot(
        data=sub,
        x="ref_len",
        hue=legend_name,
        hue_order=tip_order,
        common_norm=False,
        log_scale=True,
        ax=axes[0],
    )
    axes[0].set_ylabel("Density")
    axes[0].set_xlabel("Branch Length")
    axes[0].get_legend().remove()

    # Distribution of well and wrongly predicted branches
    for t, ls in zip(["common", "ref_unique"], ["-", "--"]):
        sns.kdeplot(
            data=sub[sub["type"] == t],
            x="ref_len",
            hue=legend_name,
            hue_order=tip_order,
            palette=pal,
            common_norm=False,
            log_scale=True,
            ls=ls,
            ax=axes[1],
        )

    # axes[1].set_ylabel("Density")
    axes[1].set_xlabel("Branch Length")
    axes[1].get_legend().remove()
    axes[1].tick_params(axis="y", which="both", left=False, labelleft=False)

    label_axes(axes, fig)

    fs = sns.plotting_context()["legend.title_fontsize"]
    lines, labels = [legend_name], [""]
    for n, col in zip(tip_order, pal):
        lines.append(Line2D([0], [0], c=col, ls="-"))
        labels.append(f"{n}")
    lines.extend(["", "Branch Inferred ?"])
    labels.extend(["", ""])
    for ls, lab in zip(["-", "--"], ["Yes", "No"]):
        lines.append(Line2D([0], [0], c="gray", ls=ls))
        labels.append(lab)
    axes[1].legend(
        lines,
        labels,
        handler_map={str: LegendTitle({"fontsize": fs})},
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full paper figure generation pipeline (slow).",
    )
    parser.add_argument(
        "--new-only",
        "--new-model-only",
        dest="new_only",
        action="store_true",
        help="With --full: only generate comparison plots for the new model (skips legacy paper plots).",
    )
    parser.add_argument(
        "--new-model-name",
        default="PF_QCLOSE+FastME",
        help="Legend label to use for the new model in plots.",
    )
    parser.add_argument(
        "--cmp-topo",
        action="append",
        default=[],
        help="LABEL=path_to_cmp_topo.csv (repeatable). If LABEL omitted, inferred from filename.",
    )
    parser.add_argument(
        "--cmp-dist",
        action="append",
        default=[],
        help="LABEL=path_to_cmp_dist.csv (repeatable). If LABEL omitted, inferred from filename.",
    )
    parser.add_argument(
        "--outdir",
        default="./figures/paper_pfbase/cmp",
        help="Output directory for cmp mode.",
    )
    parser.add_argument(
        "--dist-chunksize",
        type=int,
        default=2_000_000,
        help="Rows per chunk when reading cmp_dist.csv in cmp mode.",
    )
    parser.add_argument(
        "--output-pdf",
        default="./figures/pfbase_quartet_close/base_vs_mre_plus_qclose.pdf",
        help="Output path for single-figure mode.",
    )
    parser.add_argument(
        "--lggc-topo",
        default="./data/topos_lggc.csv",
        help="Path to topos_lggc.csv for single-figure mode.",
    )
    parser.add_argument(
        "--lggc-dist",
        default="./data/dists_lggc.csv",
        help="Path to dists_lggc.csv for single-figure mode.",
    )
    parser.add_argument(
        "--new-topo",
        "--qsiam-topo",
        dest="new_topo",
        action="append",
        default=None,
        help="Path to new-model cmp_topo.csv. Repeatable in --full/--new-model-only mode.",
    )
    parser.add_argument(
        "--new-dist",
        "--qsiam-dist",
        dest="new_dist",
        action="append",
        default=None,
        help="Path to new-model cmp_dist.csv. Repeatable in --full/--new-model-only mode.",
    )
    parser.add_argument(
        "--external-topo",
        default=None,
        help="Path to external cmp_topo.csv to include in full mode.",
    )
    parser.add_argument(
        "--external-dist",
        default=None,
        help="Path to external cmp_dist.csv to include in full mode.",
    )
    parser.add_argument(
        "--external-label",
        default=None,
        help="Method label for --external-topo/--external-dist in full mode.",
    )
    args, _ = parser.parse_known_args()
    new_paths_explicit = args.new_topo is not None or args.new_dist is not None
    new_topo_paths = _normalize_repeatable_arg(
        args.new_topo,
        "./runs/pfbase_quartet_close/eval_val_lggc/cmp_quartet_close_topo.csv",
    )
    new_dist_paths = _normalize_repeatable_arg(
        args.new_dist,
        "./runs/pfbase_quartet_close/eval_val_lggc/cmp_quartet_close_dist.csv",
    )
    if len(new_topo_paths) != len(new_dist_paths):
        raise ValueError(
            "--new-topo and --new-dist must be provided the same number of times"
        )
    if args.cmp_topo or args.cmp_dist:
        run_cmp_mode(
            args.cmp_topo, args.cmp_dist, args.outdir, args.dist_chunksize
        )
        raise SystemExit(0)

    # Set general plotting options
    sns.set_context("notebook")
    sns.set_style("darkgrid")
    new_model_label = args.new_model_name.strip() or "PF_QCLOSE+FastME"
    _register_style_alias(new_model_label)
    new_model_slug = _slugify_label(new_model_label)

    if args.new_only:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        label = outdir.name

        SAMPLING_FRAC = 0.05
        mult = 1.5
        figsize = (4 * mult + 1, 3 * mult + 1)
        figsize_all = (5 * 2 + 1, 3 * 2)
        new_results = _load_new_model_results_by_dataset(
            new_topo_paths,
            new_dist_paths,
            new_model_label,
            sample_frac=SAMPLING_FRAC,
        )

        print(f"[new-only] Saving plots to {outdir}/")
        if "lggc" in new_results:
            lggc = _load_baseline_topology("lggc")
            dists_lggc = _load_baseline_distances("lggc", SAMPLING_FRAC)
            qsiam_topo = new_results["lggc"]["topo"]
            qsiam_dists = new_results["lggc"]["dists"]
            topo_plus = pd.concat([lggc, qsiam_topo], ignore_index=True)
            dists_plus = pd.concat([dists_lggc, qsiam_dists], ignore_index=True)
            methods_plus_new_model = _methods_with_new_model(new_model_label)

            fig = base_vs_ft(
                topo_plus,
                dists_plus,
                (9, 8),
                methods=["PF+FastME", "PF_Base+FastME", new_model_label],
            )
            plt.savefig(outdir / f"{label}_base_vs_mre_plus_{new_model_slug}.pdf")
            plt.clf(); plt.cla()

            fig = single_LGGC_normRF(topo_plus, figsize, methods=methods_plus_new_model)
            plt.savefig(outdir / f"{label}_LGGC_500_rf_plus_{new_model_slug}.pdf")
            plt.clf(); plt.cla()

            fig = single_LGGC_KFscore(topo_plus, figsize, methods=methods_plus_new_model)
            plt.savefig(outdir / f"{label}_LGGC_500_kf_plus_{new_model_slug}.pdf")
            plt.clf(); plt.cla()

            fig = single_LGGC_wRF(topo_plus, figsize, methods=methods_plus_new_model)
            plt.savefig(outdir / f"{label}_LGGC_500_wrf_plus_{new_model_slug}.pdf")
            plt.clf(); plt.cla()

            fig = single_LGGC_mae(dists_plus, figsize, methods=methods_plus_new_model)
            plt.savefig(outdir / f"{label}_LGGC_500_mae_plus_{new_model_slug}.pdf")
            plt.clf(); plt.cla()

            fig = single_LGGC_mre(dists_plus, figsize, methods=methods_plus_new_model)
            plt.savefig(outdir / f"{label}_LGGC_500_mre_plus_{new_model_slug}.pdf")
            plt.clf(); plt.cla()

            fig = dataset_plot(topo_plus, dists_plus, figsize_all, methods_plus_new_model)
            plt.savefig(outdir / f"{label}_lggc_all_plus_{new_model_slug}.pdf")
            plt.clf(); plt.cla()

        for dataset_key in ["gaps", "cherry", "pastek"]:
            if dataset_key not in new_results:
                continue
            topo_base = _load_baseline_topology(dataset_key)
            dist_base = _load_baseline_distances(dataset_key, SAMPLING_FRAC)
            topo_plus = pd.concat([topo_base, new_results[dataset_key]["topo"]], ignore_index=True)
            dists_plus = pd.concat([dist_base, new_results[dataset_key]["dists"]], ignore_index=True)
            methods = _dataset_methods_with_new_model(dataset_key, new_model_label)
            prefix = _plot_filename_prefix(dataset_key)
            fig = dataset_plot(topo_plus, dists_plus, figsize_all, methods)
            plt.savefig(outdir / f"{label}_{prefix}_all_plus_{new_model_slug}.pdf")
            plt.clf(); plt.cla()

        print(f"[new-only] Done. {len(list(outdir.glob(label + '_*.pdf')))} files written.")
        raise SystemExit(0)

    if not args.full:
        generate_base_vs_mre_plus_new_model(
            output_pdf=args.output_pdf,
            lggc_topo_path=args.lggc_topo,
            lggc_dist_path=args.lggc_dist,
            new_topo_path=new_topo_paths[0],
            new_dist_path=new_dist_paths[0],
            dist_chunksize=args.dist_chunksize,
            method_label=new_model_label,
        )
        raise SystemExit(0)

    with tqdm(bar_format="[{elapsed}] {desc}", maxinterval=1) as pbar:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        ####################
        # TOPOLOGY METRICS #
        ####################

        # Parse LG+GC dataset
        pbar.set_description("Parsing LGGC topological results")
        lggc = pd.read_csv("./data/topos_lggc.csv")
        lggc["length"] = lggc["id"].apply(lambda x: x.split("_")[-1]).astype(int)
        lggc["marker"] = lggc["marker"].apply(lambda x: RENAMER.get(x, x))
        lggc["dataset"] = "LG+GC"
        pbar.update(1)

        # Choose figure size while keeping aspect ratio
        mult = 1.5
        # figsize = (4 * mult + 1, 3 * mult + 1)
        figsize = (6 * mult + 1, 3 * mult)

        # Norm RF for all aln lengths
        pbar.set_description("Plotting LG+GC topological metrics")
        fig = build_LGGC_normRF(lggc, figsize)
        plt.savefig(outdir / "combined_LGGC_rf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # KF score for all aln lengths
        fig = build_LGGC_KFscore(lggc, figsize)
        plt.savefig(outdir / "combined_LGGC_kf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # wRF score for all aln lengths
        fig = build_LGGC_wRF(lggc, figsize)
        plt.savefig(outdir / "combined_LGGC_wrf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        mult = 1.5
        figsize = (4 * mult + 1, 3 * mult + 1)

        # Single RF 500 length lGGC
        fig = single_LGGC_normRF(lggc, figsize)
        plt.savefig(outdir / "LGGC_500_rf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Single KF 500 length LGGC
        fig = single_LGGC_KFscore(lggc, figsize)
        plt.savefig(outdir / "LGGC_500_kf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Single wRF 500 length LGGC
        fig = single_LGGC_wRF(lggc, figsize)
        plt.savefig(outdir / "LGGC_500_wrf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Read fine tuned
        pbar.set_description("Parsing Pastek and Cherry Topological results")
        cherry = pd.read_csv("./data/topos_cherry.csv")
        pastek = pd.read_csv("./data/topos_pastek.csv")
        gaps = pd.read_csv("./data/topos_gaps.csv")

        for df, ds in zip([cherry, pastek, gaps], ["Cherry", "SelReg", "Indels"]):
            df["marker"] = df["marker"].apply(lambda x: RENAMER.get(x, x))
            df["dataset"], df["length"] = ds, 500

        pbar.update(1)

        # mult = 1.5
        figsize = (6 * mult + 2, 3 * mult + 1)

        # Fine tune RF
        pbar.set_description("Plotting Cherry+Pastek topological metrics")
        fig = cherry_pastek_normRF(cherry, pastek, figsize)
        plt.savefig(outdir / "cherry_pastek_rf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Fine tune KF
        fig = cherry_pastek_KFscore(cherry, pastek, figsize)
        plt.savefig(outdir / "cherry_pastek_kf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Fine tune wRF
        fig = cherry_pastek_wRF(cherry, pastek, figsize)
        plt.savefig(outdir / "cherry_pastek_wrf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Topological metrics for Cherry + Pastek
        figsize = (4.5 * mult + 1, 4.5 * mult + 1)
        fig = cherry_pastek_topos(cherry, pastek, figsize)
        plt.savefig(outdir / "cherry_pastek_topos.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # mult = 1.5
        figsize = (9 * mult + 2, 3 * mult + 1)

        # Fine tune RF
        pbar.set_description("Plotting fine-tuned topological metrics")
        fig = fine_tuned_normRF(gaps, cherry, pastek, figsize)
        plt.savefig(outdir / "fine_tune_rf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Fine tune KF
        fig = fine_tuned_KFscore(gaps, cherry, pastek, figsize)
        plt.savefig(outdir / "fine_tune_kf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Fine tune wRF
        fig = fine_tuned_wRF(gaps, cherry, pastek, figsize)
        plt.savefig(outdir / "fine_tune_wrf.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        ######################
        # EXECUTION METADATA #
        ######################

        with open("./data/model_load_times.txt", "r") as f:
            times = [float(line.strip()) for line in f]
        load_time = sum(times) / len(times)

        mult = 1.5
        figsize = (4 * mult + 1, 3 * mult + 1)

        pbar.set_description("Parsing LGGC execution metadata")
        grouped_lggc = group_elapsed("./data/execution_lggc.csv")
        grouped_lggc["marker"] = grouped_lggc["marker"].apply(
            lambda x: RENAMER.get(x, x)
        )

        pbar.update(1)

        # Memory usage 500 length LGGC
        pbar.set_description("Plotting LGGC execution metadata")
        fig = single_LGGC_elapsed(grouped_lggc, figsize, load_time)
        plt.savefig(outdir / "LGGC_500_elapsed.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Memory usage 500 length LGGC
        fig = single_LGGC_mem(grouped_lggc, figsize)
        plt.savefig(outdir / "LGGC_500_mem.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Plot Time for pastek and cherry
        pbar.set_description("Parsing fine-tuned execution metadata")
        grouped_gaps = group_elapsed("./data/execution_gaps.csv")
        grouped_cherry = group_elapsed("./data/execution_cherry.csv")
        grouped_pastek = group_elapsed("./data/execution_pastek.csv")

        grouped_gaps["marker"] = grouped_gaps["marker"].apply(
            lambda x: RENAMER.get(x, x)
        )
        grouped_cherry["marker"] = grouped_cherry["marker"].apply(
            lambda x: RENAMER.get(x, x)
        )
        grouped_pastek["marker"] = grouped_pastek["marker"].apply(
            lambda x: RENAMER.get(x, x)
        )

        pbar.update(1)

        mult = 2
        figsize = (5 * mult + 1, 3 * mult)

        # Fine tune elapsed
        pbar.set_description("Plotting fine-tuned execution metadata")
        fig = fine_tuned_elapsed(grouped_gaps, grouped_cherry, grouped_pastek, figsize)
        plt.savefig(outdir / "fine_tune_elapsed.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Fine tune mem
        fig = fine_tuned_mem(grouped_gaps, grouped_cherry, grouped_pastek, figsize)
        plt.savefig(outdir / "fine_tune_mem.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Mixed dataset execution time
        mult = 1.5
        figsize = (4 * mult + 1, 3 * mult + 1)
        fig = paper_elapsed(
            pd.concat(
                [
                    grouped_lggc[grouped_lggc["length"] == 500],
                    grouped_cherry[grouped_cherry["marker"] == "IQTree_MF"],
                ]
            ),
            figsize,
        )
        plt.savefig(outdir / "elapsed.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        fig = paper_elapsed(
            pd.concat(
                [
                    grouped_lggc[grouped_lggc["length"] == 500],
                    grouped_cherry[grouped_cherry["marker"] == "IQTree_MF"],
                ]
            ),
            figsize,
            load_time
        )
        plt.savefig(outdir / "elapsed_pf_loads.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        ###########
        # MAE     #
        ###########

        # Percent of pairwise distances to plot
        SAMPLING_FRAC = 0.05

        mult = 1.5
        figsize = (4 * mult + 1, 3 * mult + 1)

        pbar.set_description("Parsing LGGC distance results")
        dists_lggc = pd.read_csv("./data/dists_lggc.csv").sample(frac=SAMPLING_FRAC)
        dists_lggc["n_tips"] = (
            dists_lggc["id"].apply(lambda x: x.split("_")[1]).astype(int)
        )
        dists_lggc["length"] = (
            dists_lggc["id"].apply(lambda x: x.split("_")[-1]).astype(int)
        )
        dists_lggc["MAE"] = (dists_lggc["ref_dist"] - dists_lggc["cmp_dist"]).abs()
        dists_lggc["MRE"] = dists_lggc["MAE"] / dists_lggc["ref_dist"]
        dists_lggc["MRD"] = (
            dists_lggc["ref_dist"] - dists_lggc["cmp_dist"]
        ) / dists_lggc["ref_dist"]
        dists_lggc["marker"] = dists_lggc["marker"].apply(lambda x: RENAMER.get(x, x))
        dists_lggc["dataset"] = "LG+GC"
        pbar.update(1)

        # MRE for aln 500 LG+GC
        pbar.set_description("Plotting LGGC distance results")
        fig = single_LGGC_mre(dists_lggc, figsize)
        plt.savefig(outdir / "LGGC_500_mre.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # MAE for aln 500 LG+GC
        fig = single_LGGC_mae(dists_lggc, figsize)
        plt.savefig(outdir / "LGGC_500_mae.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Quantile plots
        sub = dists_lggc[dists_lggc["length"] == 500]
        sub["percentile"] = pd.qcut(sub["ref_dist"], 100).apply(lambda x: x.right)

        # Bin distances logarithmically
        bins = np.logspace(
            np.trunc(np.log10(sub["ref_dist"].min())),
            np.ceil(np.log10(sub["ref_dist"].max())),
            100,
        )
        sub["binned"] = pd.cut(sub["ref_dist"], bins=bins).apply(lambda x: x.right)

        # Distance percentile vs MAE
        single_LGGC_quantiles_mae(sub, figsize)
        plt.savefig(outdir / "LGGC_500_quantile_mae.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Distance percentile vs MRE
        single_LGGC_quantiles_mre(sub, figsize)
        plt.savefig(outdir / "LGGC_500_quantile_mre.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Distance percentile vs MRE
        single_LGGC_quantiles_mrd(sub, figsize)
        plt.savefig(outdir / "LGGC_500_quantile_mrd.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Binned Distance vs MAE
        single_LGGC_binned_mae(sub, figsize)
        plt.savefig(outdir / "LGGC_500_binned_mae.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Binned Distance vs MRE
        single_LGGC_binned_mre(sub, figsize)
        plt.savefig(outdir / "LGGC_500_binned_mre.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Binned Distance vs MRE
        single_LGGC_binned_mrd(sub, figsize)
        plt.savefig(outdir / "LGGC_500_binned_mrd.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        # Distribution of pairwise distances in test set trees
        legend_name = "Number of leaves"
        sub = dists_lggc[
            (dists_lggc["length"] == 500) & (dists_lggc["marker"] == "PF+FastME")
        ].rename({"n_tips": legend_name}, axis=1)
        tip_order = sorted(sub[legend_name].unique())
        fig, ax = plt.subplots(1, layout="constrained")
        sns.kdeplot(
            data=sub,
            x="ref_dist",
            hue=legend_name,
            hue_order=tip_order,
            common_norm=False,
            log_scale=True,
            ax=ax,
        )
        ax.set_ylabel("Density")
        ax.set_xlabel("Pairwise Distance")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.savefig(outdir / "pairwise_dist_testset.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        fig = base_vs_ft(
            lggc[lggc["length"] == 500], dists_lggc[dists_lggc["length"] == 500], (9, 8)
        )
        plt.savefig(outdir / "base_vs_mre.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        new_model_results = _load_new_model_results_by_dataset(
            new_topo_paths,
            new_dist_paths,
            new_model_label,
            sample_frac=SAMPLING_FRAC,
            skip_missing=not new_paths_explicit,
        )
        qsiam_topo = None
        qsiam_dists = None
        if "lggc" in new_model_results:
            pbar.set_description(f"Parsing {new_model_label} evaluation results")
            qsiam_topo = new_model_results["lggc"]["topo"]
            qsiam_dists = new_model_results["lggc"]["dists"]
            pbar.update(1)
            methods_plus_new_model = _methods_with_new_model(new_model_label)

            pbar.set_description(f"Plotting LG+GC + {new_model_label} figures")
            topo_plus = pd.concat(
                [lggc[lggc["length"] == 500], qsiam_topo], ignore_index=True
            )
            dists_plus = pd.concat(
                [dists_lggc[dists_lggc["length"] == 500], qsiam_dists], ignore_index=True
            )

            fig = base_vs_ft(
                topo_plus,
                dists_plus,
                (9, 8),
                methods=["PF+FastME", "PF_Base+FastME", new_model_label],
            )
            plt.savefig(outdir / f"base_vs_mre_plus_{new_model_slug}.pdf")
            plt.clf()
            plt.cla()
            pbar.update(1)

            fig = single_LGGC_normRF(
                topo_plus, figsize, methods=methods_plus_new_model
            )
            plt.savefig(outdir / f"LGGC_500_rf_plus_{new_model_slug}.pdf")
            plt.clf()
            plt.cla()
            pbar.update(1)

            fig = single_LGGC_KFscore(
                topo_plus, figsize, methods=methods_plus_new_model
            )
            plt.savefig(outdir / f"LGGC_500_kf_plus_{new_model_slug}.pdf")
            plt.clf()
            plt.cla()
            pbar.update(1)

            fig = single_LGGC_wRF(
                topo_plus, figsize, methods=methods_plus_new_model
            )
            plt.savefig(outdir / f"LGGC_500_wrf_plus_{new_model_slug}.pdf")
            plt.clf()
            plt.cla()
            pbar.update(1)

            fig = single_LGGC_mae(
                dists_plus, figsize, methods=methods_plus_new_model
            )
            plt.savefig(outdir / f"LGGC_500_mae_plus_{new_model_slug}.pdf")
            plt.clf()
            plt.cla()
            pbar.update(1)

            fig = single_LGGC_mre(
                dists_plus, figsize, methods=methods_plus_new_model
            )
            plt.savefig(outdir / f"LGGC_500_mre_plus_{new_model_slug}.pdf")
            plt.clf()
            plt.cla()
            pbar.update(1)

        external_topo = None
        external_dists = None
        external_label = args.external_label.strip() if args.external_label else None
        if args.external_topo or args.external_dist or external_label:
            if not (args.external_topo and args.external_dist and external_label):
                raise ValueError(
                    "--external-topo, --external-dist, and --external-label must be provided together"
                )
            external_topo_path = Path(args.external_topo)
            external_dist_path = Path(args.external_dist)
            if external_topo_path.exists() and external_dist_path.exists():
                pbar.set_description(f"Parsing {external_label} evaluation results")
                external_topo, external_dists = _load_external_results(
                    external_topo_path,
                    external_dist_path,
                    external_label,
                    default_length=500,
                )
                pbar.update(1)

                external_slug = re.sub(r"[^a-z0-9]+", "_", external_label.lower()).strip("_")
                methods_plus_external = sorted(LGGC_METHODS_NO_HAMMING + [external_label])

                pbar.set_description(f"Plotting LG+GC + {external_label} figures")
                topo_plus = pd.concat(
                    [lggc[lggc["length"] == 500], external_topo], ignore_index=True
                )
                dists_plus = pd.concat(
                    [dists_lggc[dists_lggc["length"] == 500], external_dists],
                    ignore_index=True,
                )

                fig = base_vs_ft(
                    topo_plus,
                    dists_plus,
                    (9, 8),
                    methods=["PF+FastME", "PF_Base+FastME", external_label],
                )
                plt.savefig(outdir / f"base_vs_mre_plus_{external_slug}.pdf")
                plt.clf()
                plt.cla()
                pbar.update(1)

                fig = single_LGGC_normRF(
                    topo_plus, figsize, methods=methods_plus_external
                )
                plt.savefig(outdir / f"LGGC_500_rf_plus_{external_slug}.pdf")
                plt.clf()
                plt.cla()
                pbar.update(1)

                fig = single_LGGC_KFscore(
                    topo_plus, figsize, methods=methods_plus_external
                )
                plt.savefig(outdir / f"LGGC_500_kf_plus_{external_slug}.pdf")
                plt.clf()
                plt.cla()
                pbar.update(1)

                fig = single_LGGC_wRF(
                    topo_plus, figsize, methods=methods_plus_external
                )
                plt.savefig(outdir / f"LGGC_500_wrf_plus_{external_slug}.pdf")
                plt.clf()
                plt.cla()
                pbar.update(1)

                fig = single_LGGC_mae(
                    dists_plus, figsize, methods=methods_plus_external
                )
                plt.savefig(outdir / f"LGGC_500_mae_plus_{external_slug}.pdf")
                plt.clf()
                plt.cla()
                pbar.update(1)

                fig = single_LGGC_mre(
                    dists_plus, figsize, methods=methods_plus_external
                )
                plt.savefig(outdir / f"LGGC_500_mre_plus_{external_slug}.pdf")
                plt.clf()
                plt.cla()
                pbar.update(1)

        mult = 2
        figsize = (5 * mult + 1, 3 * mult)
        # Fine tune MAE
        pbar.set_description("Parsing fine-tuned distance results")
        dists_cherry = pd.read_csv("./data/dists_cherry.csv").sample(frac=SAMPLING_FRAC)
        dists_pastek = pd.read_csv("./data/dists_pastek.csv").sample(frac=SAMPLING_FRAC)
        dists_gaps = pd.read_csv("./data/dists_gaps.csv").sample(frac=SAMPLING_FRAC)
        for df, ds in zip(
            [dists_cherry, dists_pastek, dists_gaps], ["Cherry", "SelReg", "Indels"]
        ):
            df["marker"] = df["marker"].apply(lambda x: RENAMER.get(x, x))
            df["n_tips"] = df["id"].apply(lambda x: x.split("_")[1]).astype(int)
            df["MAE"] = (df["ref_dist"] - df["cmp_dist"]).abs()
            df["MRE"] = df["MAE"] / df["ref_dist"]
            df["dataset"], df["length"] = ds, 500  # Needed for mis-specification plots
        pbar.update(1)

        pbar.set_description("Plotting fine-tuned distance results")
        fig = fine_tuned_mae(dists_gaps, dists_cherry, dists_pastek, figsize)
        plt.savefig(outdir / "fine_tune_mae.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        figsize = (6.5, 6)
        pbar.set_description("Plotting Histograms on distances in LGGC")
        fig = hist_LGGC(dists_lggc[dists_lggc["length"] == 500], figsize)
        plt.savefig(outdir / "dist_hist_LGGC.png", dpi=150)
        plt.clf()
        plt.cla()
        pbar.update(1)

        # figsize = (10.5, 15)
        pbar.set_description("Plotting Histograms on distances in Pastek")
        fig = hist_cherry_4x4(dists_cherry, figsize)
        plt.savefig(outdir / "dist_hist_cherry.png", dpi=150)
        plt.clf()
        plt.cla()
        pbar.update(1)

        pbar.set_description("Plotting Histograms on distances in Pastek")
        fig = hist_pastek_4x4(dists_pastek, figsize)
        plt.savefig(outdir / "dist_hist_pastek.png", dpi=150)
        plt.clf()
        plt.cla()
        pbar.update(1)

        ###############
        # ALL GROUPED #
        ###############

        mult = 2
        figsize = (5 * mult + 1, 3 * mult)

        pbar.set_description("Plotting all metrics for LG+GC")
        fig = dataset_plot(
            lggc[lggc["length"] == 500],
            dists_lggc[dists_lggc["length"] == 500],
            figsize,
            LGGC_METHODS_NO_HAMMING,
        )
        plt.savefig(outdir / "lggc_all.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        if qsiam_topo is not None and qsiam_dists is not None:
            pbar.set_description(f"Plotting all metrics for LG+GC + {new_model_label}")
            fig = dataset_plot(
                pd.concat([lggc[lggc["length"] == 500], qsiam_topo], ignore_index=True),
                pd.concat(
                    [dists_lggc[dists_lggc["length"] == 500], qsiam_dists],
                    ignore_index=True,
                ),
                figsize,
                methods_plus_new_model,
            )
            plt.savefig(outdir / f"lggc_all_plus_{new_model_slug}.pdf")
            plt.clf()
            plt.cla()
            pbar.update(1)

        if external_topo is not None and external_dists is not None and external_label is not None:
            external_slug = re.sub(r"[^a-z0-9]+", "_", external_label.lower()).strip("_")
            pbar.set_description(f"Plotting all metrics for LG+GC + {external_label}")
            fig = dataset_plot(
                pd.concat(
                    [lggc[lggc["length"] == 500], external_topo], ignore_index=True
                ),
                pd.concat(
                    [dists_lggc[dists_lggc["length"] == 500], external_dists],
                    ignore_index=True,
                ),
                figsize,
                sorted(LGGC_METHODS_NO_HAMMING + [external_label]),
            )
            plt.savefig(outdir / f"lggc_all_plus_{external_slug}.pdf")
            plt.clf()
            plt.cla()
            pbar.update(1)

        pbar.set_description("Plotting all metrics for cherry")
        fig = dataset_plot(
            cherry,
            dists_cherry,
            figsize,
            sorted(sorted(["IQTree_LG+GC"] + FINE_TUNE_METHODS) + ["PF_Cherry+FastME"]),
        )
        plt.savefig(outdir / "cherry_all.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        pbar.set_description("Plotting all metrics for pastek")
        fig = dataset_plot(
            pastek,
            dists_pastek,
            figsize,
            sorted(sorted(["IQTree_LG+GC"] + FINE_TUNE_METHODS) + ["PF_SelReg+FastME"]),
        )
        plt.savefig(outdir / "pastek_all.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        pbar.set_description("Plotting all metrics for gaps")
        fig = dataset_plot(
            gaps,
            dists_gaps,
            figsize,
            # Removing PF_MRE because it crushes everything
            sorted(
                [x for x in LGGC_METHODS_NO_HAMMING if x != "PF+FastME"]
                + ["PF_Indel+FastME"]
            ),
        )
        plt.savefig(outdir / "gaps_all.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        for dataset_key, topo_base, dist_base in [
            ("cherry", cherry, dists_cherry),
            ("pastek", pastek, dists_pastek),
            ("gaps", gaps, dists_gaps),
        ]:
            if dataset_key not in new_model_results:
                continue
            prefix = _plot_filename_prefix(dataset_key)
            pbar.set_description(
                f"Plotting all metrics for {prefix} + {new_model_label}"
            )
            fig = dataset_plot(
                pd.concat(
                    [topo_base, new_model_results[dataset_key]["topo"]],
                    ignore_index=True,
                ),
                pd.concat(
                    [dist_base, new_model_results[dataset_key]["dists"]],
                    ignore_index=True,
                ),
                figsize,
                _dataset_methods_with_new_model(dataset_key, new_model_label),
            )
            plt.savefig(outdir / f"{prefix}_all_plus_{new_model_slug}.pdf")
            plt.clf()
            plt.cla()
            pbar.update(1)

        # Model mis-specification plots
        PFS = ["PF+FastME", "PF_Indel+FastME", "PF_Cherry+FastME", "PF_SelReg+FastME"]

        # Subset and collect PF results
        dfs = []
        for df in [lggc, cherry, gaps, pastek]:
            dfs.append(df[(df["marker"].isin(PFS)) & (df["length"] == 500)])
        topo = pd.concat(dfs)
        dfs = []
        for df in [dists_lggc, dists_cherry, dists_gaps, dists_pastek]:
            dfs.append(df[(df["marker"].isin(PFS)) & (df["length"] == 500)])
        dists = pd.concat(dfs)
        for df in [topo, dists]:
            df["marker"] = df["marker"].apply(lambda x: x.removesuffix("+FastME"))

        means_alltips = (
            topo.groupby(["dataset", "marker"])[["norm_rf", "kf_score"]]
            .mean()
            .join(dists.groupby(["dataset", "marker"])[["MAE", "MRE"]].mean())
        )

        means_50 = (
            topo[topo["n_tips"] == 50]
            .groupby(["dataset", "marker"])[["norm_rf", "kf_score"]]
            .mean()
            .join(
                dists[dists["n_tips"] == 50]
                .groupby(["dataset", "marker"])[["MAE", "MRE"]]
                .mean()
            )
        )

        datasets = ["LG+GC", "Indels", "Cherry", "SelReg"]
        pf_order = [x.removesuffix("+FastME") for x in PFS]

        mult = 1.5
        figsize = (mult * 5, mult * 5)

        # 50 tips plot
        fig = misspecification(means_50, figsize, datasets, pf_order)
        plt.savefig(outdir / "misspecification_50tips.pdf")
        plt.clf()
        plt.cla()

        # All tips plot
        fig = misspecification(means_alltips, figsize, datasets, pf_order)
        plt.savefig(outdir / "misspecification_alltips.pdf")
        plt.clf()
        plt.cla()

        ##############
        # LIKELIHOOD #
        ##############

        pbar.set_description("Reading and plotting LGGC likelihoods")
        lik_lggc = pd.read_csv("./data/likelihoods_lggc.csv")
        lik_lggc["marker"] = lik_lggc["marker"].apply(lambda x: RENAMER.get(x, x))

        mult = 1.5
        figsize = (6 * mult + 1, 3 * mult)

        # Norm RF for all aln lengths
        fig = build_LGGC_lik(lik_lggc, figsize)
        plt.savefig(outdir / "combined_LGGC_lik.pdf")
        plt.clf()
        plt.cla()

        mult = 1.5
        figsize = (4 * mult + 1, 3 * mult + 1)

        # Single RF 500 length lGGC
        fig = single_LGGC_lik(lik_lggc, figsize)
        plt.savefig(outdir / "LGGC_500_lik.pdf")
        plt.clf()
        plt.cla()
        pbar.update(1)

        ##################
        # BRANCH LENGTHS #
        ##################

        brlens = pd.read_csv("./data/brlens_lggc.csv")
        legend_name = "Number of leaves"
        brlens["length"] = brlens["id"].apply(lambda x: x.split("_")[-1]).astype(int)
        brlens[legend_name] = brlens["id"].apply(lambda x: x.split("_")[1]).astype(int)
        brlens["type"] = (
            brlens["ref_len"].isna() * 10 + brlens["cmp_len"].isna()
        ).apply({0: "common", 1: "ref_unique", 10: "cmp_unique"}.get)
        sub = brlens[(brlens["length"] == 500) & (brlens["marker"] == "PF+FastME")]

        fig = plot_brlen_dists(sub, (10, 4))
        plt.savefig(outdir / "branch_length_errors.pdf")
        plt.savefig(outdir / "branch_length_errors.svg")
        plt.clf()
        plt.cla()
        pbar.update(1)
