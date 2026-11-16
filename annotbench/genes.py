"""Validate gene identifiers against NCBI gene_info while keeping the originals.

Nothing is dropped or merged here. Each feature is labelled with how it was
resolved, so the preprocessing step can apply an explicit policy later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HUMAN_TAX_ID = "9606"


@dataclass(frozen=True)
class GeneReference:
    symbol_to_id: dict[str, str]
    synonym_to_id: dict[str, str]
    ambiguous_names: frozenset[str]
    mitochondrial_ids: frozenset[str]

    @property
    def known_names(self) -> set[str]:
        return set(self.symbol_to_id) | set(self.synonym_to_id)


def _split_unique(keys: pd.Series, ids: pd.Series) -> tuple[dict[str, str], set[str]]:
    """Return (name -> GeneID for names with exactly one GeneID, names with several)."""
    pairs = pd.DataFrame({"key": keys, "id": ids}).dropna().drop_duplicates()
    repeated = pairs["key"].duplicated(keep=False)
    unique = pairs[~repeated]
    return dict(zip(unique["key"], unique["id"])), set(pairs.loc[repeated, "key"])


def load_gene_reference(path: str | Path) -> GeneReference:
    columns = ["#tax_id", "GeneID", "Symbol", "Synonyms", "chromosome"]
    info = pd.read_csv(path, sep="\t", usecols=columns, dtype=str)
    # The human file also holds a few Neanderthal and Denisovan rows.
    info = info[info["#tax_id"] == HUMAN_TAX_ID]

    symbol_to_id, ambiguous_symbols = _split_unique(info["Symbol"], info["GeneID"])
    synonyms = info[info["Synonyms"] != "-"].copy()
    synonyms["synonym"] = synonyms["Synonyms"].str.split("|")
    synonyms = synonyms.explode("synonym")
    # A synonym that is another gene's official symbol would be a misleading match.
    synonyms = synonyms[~synonyms["synonym"].isin(info["Symbol"])]
    synonym_to_id, ambiguous_synonyms = _split_unique(synonyms["synonym"], synonyms["GeneID"])

    return GeneReference(
        symbol_to_id=symbol_to_id,
        synonym_to_id=synonym_to_id,
        ambiguous_names=frozenset(ambiguous_symbols | ambiguous_synonyms),
        mitochondrial_ids=frozenset(info.loc[info["chromosome"] == "MT", "GeneID"]),
    )


def repair_r_names(names: pd.Series, known_names: set[str]) -> tuple[pd.Series, pd.Series]:
    """Undo R's make.names only when NCBI confirms the repaired name exists."""
    candidate = names.str.replace(".", "-", regex=False)
    accepted = names.str.contains(".", regex=False) & candidate.isin(known_names)
    return candidate.where(accepted, names), accepted


def resolve_features(
    feature_ids: pd.Index,
    source_symbols: pd.Series,
    reference: GeneReference,
    *,
    r_mangled_names: bool,
    spike_in: pd.Series,
) -> pd.DataFrame:
    """Return one row per feature describing how its identifier was validated."""
    original = pd.Series(np.asarray(source_symbols, dtype=object), index=feature_ids, dtype="string")
    if r_mangled_names:
        symbol, repaired = repair_r_names(original, reference.known_names)
    else:
        symbol, repaired = original, pd.Series(False, index=feature_ids)

    is_spike = pd.Series(np.asarray(spike_in, dtype=bool), index=feature_ids)
    official = symbol.map(reference.symbol_to_id)
    synonym = symbol.map(reference.synonym_to_id)
    gene_id = official.fillna(synonym).astype("string").mask(is_spike)

    how = pd.Series("unmatched", index=feature_ids, dtype="string")
    how[symbol.isin(reference.ambiguous_names)] = "ambiguous_name"
    how[synonym.notna()] = "unique_synonym"
    how[official.notna()] = "official_symbol"
    how[is_spike] = "spike_in"

    return pd.DataFrame(
        {
            "feature_id_original": feature_ids.astype(str),
            "symbol_original": original,
            "symbol": symbol,
            "symbol_repaired": repaired.astype(bool),
            "ncbi_gene_id": gene_id,
            "id_resolution": how,
            "is_spike_in": is_spike,
            "is_mt": symbol.str.startswith("MT-").fillna(False).astype(bool)
            | gene_id.isin(reference.mitochondrial_ids),
            "shares_gene_id": gene_id.notna() & gene_id.duplicated(keep=False),
        },
        index=feature_ids,
    )
