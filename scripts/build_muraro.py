"""Build the Muraro AnnData file from the GEO deposit and the ExperimentHub labels.

This is build_muraro() from data_preview_dataset_B_and_C.ipynb moved into a
script, with paths taken from configs/datasets.yaml instead of notebook globals.

Usage, from the repo root:
    python scripts/build_muraro.py [--data-root DIR] [--out FILE] [--force]
"""
import argparse
import sys
from pathlib import Path

import anndata as ad
import GEOparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from annotbench import config  # noqa: E402


def build(root: Path, cfg: dict) -> ad.AnnData:
    inputs = cfg["build_inputs"]
    k = cfg["umi_back_transform"]["k"]

    table = pd.read_csv(config.require_file(root, inputs["matrix"], "Muraro GEO matrix"), sep="\t", index_col=0)
    var = pd.DataFrame(index=table.index.astype(str))
    parts = var.index.str.partition("__")
    var["symbol"], var["chromosome"] = parts.get_level_values(0), parts.get_level_values(2)
    var["is_ercc"] = var["symbol"].str.startswith("ERCC-")
    var.loc[var["is_ercc"], "chromosome"] = "spike-in"
    var.index.name = "gene_id"

    obs = pd.DataFrame(index=table.columns.astype(str))
    obs.index.name = "cell_id"
    obs["library"] = obs.index.str.split("_").str[0]
    obs["donor"] = obs["library"].str.split("-").str[0]
    obs["well"] = obs.index.str.split("_").str[1].astype(int)

    GEOparse.set_verbosity("ERROR")
    series = GEOparse.get_GEO(filepath=str(config.require_file(root, inputs["soft"], "Muraro SOFT file")))
    library_to_gsm = {"; ".join(gsm.metadata.get("description", [])): name for name, gsm in series.gsms.items()}
    obs["gsm"] = obs["library"].map(library_to_gsm)
    barcodes = pd.read_csv(
        config.require_file(root, inputs["barcodes"], "Muraro barcode table"), sep="\t", header=None, names=["well", "barcode"]
    ).set_index("well")["barcode"]
    obs["barcode"] = obs["well"].map(barcodes)
    if obs["gsm"].isna().any() or obs["barcode"].isna().any():
        raise ValueError("some wells have no GEO sample or barcode")

    labels = pd.read_csv(config.require_file(root, inputs["labels"], "Muraro label table"), index_col="cell_id")
    if set(labels.index) != set(obs.index):
        raise ValueError("label table and GEO wells do not match one to one")
    labels = labels.loc[obs.index]
    if not (labels["donor"] == obs["donor"]).all():
        raise ValueError("donor mismatch between label table and GEO")
    obs["cell_type1"] = labels["label"]
    obs["author_qc_pass"] = obs["cell_type1"].notna()

    transcripts = sp.csr_matrix(table.values.T, dtype="float32")
    # The deposit stores t = -K ln(1 - u/K). Inverting it recovers the observed UMI count u.
    umi = transcripts.copy()
    umi.data = np.round(k * (1 - np.exp(-umi.data.astype("float64") / k))).astype("float32")

    adata = ad.AnnData(X=transcripts, obs=obs, var=var, layers={"transcripts": transcripts.copy(), "umi": umi})
    adata.uns["X_source_assay"] = "transcripts"
    adata.uns["source_files"] = [inputs["soft"], inputs["matrix"], inputs["barcodes"], inputs["labels"]]
    return adata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--registry", default=str(config.DEFAULT_REGISTRY))
    parser.add_argument("--out", default=None, help="output file (default: the path in the registry)")
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    args = parser.parse_args()

    root = config.resolve_data_root(args.data_root)
    cfg = config.dataset_config(config.load_registry(args.registry), "muraro")
    out = Path(args.out) if args.out else root / cfg["h5ad"]
    if out.exists() and not args.force:
        print(f"{out} already exists. Pass --force to replace it.", file=sys.stderr)
        return 2
    adata = build(root, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out, compression="gzip")
    print(f"Wrote {out}: {adata.n_obs} wells x {adata.n_vars} features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
