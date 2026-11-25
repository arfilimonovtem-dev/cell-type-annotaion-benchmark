"""The split step: give every reference cell exactly one role per scenario, from metadata only.

Plain-words picture: deal a box of flashcards into piles before class starts.
``train`` is the study pile, ``calibration`` the practice quiz, ``evaluation``
the final exam. Cards never move between piles, and whole *families* (donors)
go into one pile, because cells from one donor resemble each other.

Two files per scenario (decided 2026-11-23):

* The **split** covers the *reference* dataset only (Baron). It is the only
  place randomness enters (the seeded cap). In a within-dataset scenario (E1, or
  E3 within Baron) the held-out donor's cells are in the split too, as
  ``evaluation`` / ``evaluation_open_set``, because they are reference-dataset cells.
* The **query set** (``make_query_set``) is used when the query is another
  dataset (Muraro). The query dataset never enters the split: none of its cells
  can be trained on, calibrated on or capped. The query set only records which
  query cells are scored, and how: ``evaluation`` (a type the reference trained
  on), ``evaluation_open_set`` (a type it did not), ``challenge_unlabeled`` or
  ``excluded``. It has no seed, because nothing in it is random.

Hard rules this module enforces (proposal IV.E, VI.A-C, VII.C; docs/io_spec.md §5, §8):

* Roles are decided from donors, labels, QC verdicts and a seed. Expression
  values are never read: ``make_split`` receives ``obs`` only.
* No donor gives cells to both ``train`` and ``calibration``.
* In a within-dataset scenario the evaluation donors give no cell to any reference role.
* A removed class (E3) leaves ``train``, ``train_reserve`` and ``calibration``;
  its query cells stay, as ``evaluation_open_set``.
* Query-dataset cells never appear in a split file.
* The 500-per-class cap is a seeded draw over *sorted* cell IDs, so the same seed
  gives the same cells whatever order the rows arrive in.
* Under-supported classes are reported, never fixed by redrawing the split.

What is deliberately left to configuration (decisions, not code):
which donors are calibration / evaluation donors, the cap, the support
thresholds, what to do with under-supported classes, and whether the reference
keeps classes the query does not have.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import read_tsv, sha256_file, write_tsv

ROLES = (
    "train", "train_reserve", "calibration", "evaluation", "evaluation_open_set",
    "challenge_unlabeled", "excluded",
)
REFERENCE_ROLES = ("train", "train_reserve", "calibration")
QUERY_ROLES = ("evaluation", "evaluation_open_set", "challenge_unlabeled")
SPLIT_COLUMNS = [
    "dataset", "cell_id", "donor", "label_original", "label_harmonized", "role", "role_reason", "inner_fold",
]
QUERY_COLUMNS = SPLIT_COLUMNS[:-1]   # a query set has no inner folds: nothing in it is trained on
REQUIRED_OBS = [
    "dataset", "donor", "label_original", "label_harmonized", "eligible_primary", "exclusion_reason",
    "qc_pass", "qc_reason",
]
FIELD_VALUE = re.compile(r"^[a-z0-9_.]+$")  # io_spec §3: '__' must always separate fields


class SplitError(ValueError):
    """A split breaks one of the rules above. The message lists every problem found."""


@dataclass(frozen=True)
class Scenario:
    """One experiment definition, without method and seed.

    E1 (within-dataset): reference == query; ``eval_donors`` are the exam family.
    E2 (transfer):       reference != query; every usable query cell is examined.
    E3 (class removal):  like E2 (cross-dataset) or E1 (within-dataset) plus ``remove``.
                         The within-dataset version is the matched control H1 needs (VI.C).
    """

    exp: str
    reference: str
    query: str
    calibration_donors: tuple[str, ...]
    eval_donors: tuple[str, ...] = ()
    remove: tuple[str, ...] = ()
    label_task: str = "pancreas_harmonized_v1"
    # "all": the reference trains on every eligible class it has (realistic: a
    # method does not know what the query contains). "shared": only classes the
    # query also has (a closed-set task). DECISION (docs/io_spec.md §7 item 5).
    train_classes: str = "all"

    @classmethod
    def from_config(cls, entry: dict, label_task: str) -> "Scenario":
        known = {"exp", "reference", "query", "calibration_donors", "eval_donors", "remove", "train_classes"}
        unknown = sorted(set(entry) - known)
        if unknown:
            raise SplitError(f"scenario {entry}: unknown keys {unknown}")
        scenario = cls(
            exp=entry["exp"],
            reference=entry["reference"],
            query=entry["query"],
            calibration_donors=tuple(entry.get("calibration_donors") or ()),
            eval_donors=tuple(entry.get("eval_donors") or ()),
            remove=tuple(entry.get("remove") or ()),
            label_task=label_task,
            train_classes=entry.get("train_classes", "all"),
        )
        scenario.validate()
        return scenario

    def validate(self) -> None:
        problems = []
        if self.exp not in {"e1", "e2", "e3"}:
            problems.append(f"exp '{self.exp}' is not implemented (pilot supports e1, e2, e3)")
        within = self.reference == self.query
        if within and not self.eval_donors:
            problems.append("within-dataset scenario needs eval_donors (the held-out exam donors)")
        if not within and self.eval_donors:
            problems.append("eval_donors only make sense when reference == query")
        if not self.calibration_donors:
            problems.append("calibration_donors is empty: calibration cells must come from named donors")
        if set(self.calibration_donors) & set(self.eval_donors):
            problems.append("a donor cannot be both a calibration and an evaluation donor")
        if self.exp == "e3" and not self.remove:
            problems.append("e3 needs a class in 'remove'")
        if self.exp != "e3" and self.remove:
            problems.append("'remove' is only used by e3")
        if len(self.remove) > 1:
            problems.append("the pilot supports removing one class per scenario")
        if self.train_classes not in {"all", "shared"}:
            problems.append("train_classes must be 'all' or 'shared'")
        for value in (self.reference, self.query, self.label_task, *self.eval_donors, *self.remove):
            if not FIELD_VALUE.match(value):
                problems.append(f"'{value}' contains characters not allowed in a scenario ID")
        if problems:
            raise SplitError("invalid scenario: " + "; ".join(problems))

    @property
    def perturbation(self) -> str:
        return "remove." + self.remove[0] if self.remove else "none"

    @property
    def query_token(self) -> str:
        """io_spec §3 example: ``qry-baron.human3`` names the held-out donor in E1."""
        if self.reference == self.query:
            return ".".join([self.query, *sorted(self.eval_donors)])
        return self.query

    @property
    def within(self) -> bool:
        return self.reference == self.query

    @property
    def scenario_id(self) -> str:
        """The split ID without the seed: names the query set, which no seed changes."""
        return (f"exp-{self.exp}__ref-{self.reference}__qry-{self.query_token}__task-{self.label_task}"
                f"__pert-{self.perturbation}")

    def split_id(self, seed: int) -> str:
        return f"{self.scenario_id}__seed-{seed:02d}"


@dataclass(frozen=True)
class SplitRules:
    cap_per_class: int | None = 500           # proposal VI.B
    min_calibration_per_class: int = 20       # proposal VII.C
    min_train_per_class: int = 20             # suggested, io_spec §7 item 4 (DECISION)
    min_eval_per_class: int = 10              # suggested, io_spec §7 item 4 (DECISION)
    donor_dominance: float = 0.9              # flag when one donor gives >= 90% of a class's training cells
    challenge_reasons: tuple[str, ...] = ("uncertain_label", "suspected_doublet")  # proposal IV.A
    # "flag": keep under-supported training classes and report them.
    # "restrict": drop them from every reference role; their query cells become open-set.
    support_policy: str = "flag"

    @classmethod
    def from_config(cls, cfg: dict | None) -> "SplitRules":
        cfg = dict(cfg or {})
        if "challenge_reasons" in cfg:
            cfg["challenge_reasons"] = tuple(cfg["challenge_reasons"])
        rules = cls(**cfg)
        if rules.support_policy not in {"flag", "restrict"}:
            raise SplitError("support_policy must be 'flag' or 'restrict'")
        return rules


def seeded_rng(seed: int, purpose: str, *keys: str) -> np.random.Generator:
    """One independent generator per (seed, purpose, key).

    Deriving the generator from the purpose ("cap") and the class name means that
    adding a new random step later, or a new class, never changes the cap draw of
    an existing class. crc32 is used because Python's own hash() of a string is
    randomised per process.
    """
    entropy = [int(seed), zlib.crc32(purpose.encode())] + [zlib.crc32(str(k).encode()) for k in keys]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("")


def _metadata(obs: pd.DataFrame, dataset: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_OBS if c not in obs.columns]
    if missing:
        raise SplitError(f"obs lacks columns {missing}. Run attach_labels and QC first.")
    df = obs.loc[obs["dataset"] == dataset, REQUIRED_OBS].copy()
    if df.empty:
        raise SplitError(f"no cells of dataset '{dataset}' in obs")
    df.insert(1, "cell_id", df.index.astype(str))
    df = df.reset_index(drop=True)
    if df["cell_id"].duplicated().any():
        raise SplitError(f"duplicate cell IDs: {df.loc[df['cell_id'].duplicated(), 'cell_id'].head().tolist()}")
    for column in ("dataset", "donor", "label_original", "label_harmonized", "exclusion_reason", "qc_reason"):
        df[column] = _text(df[column])
    return df


def _unusable(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(usable, unusable, reason) from author exclusions and QC verdicts."""
    eligible = df["eligible_primary"].astype(bool)
    qc_pass = df["qc_pass"].astype(bool)
    qc_text = ("qc_fail:" + df["qc_reason"]).where(~qc_pass, "")
    combined = (df["exclusion_reason"] + ";" + qc_text).str.strip(";")
    usable = eligible & qc_pass
    return usable, ~usable, combined


def make_split(obs: pd.DataFrame, scenario: Scenario, rules: SplitRules, seed: int,
               query_classes: set[str] | None = None) -> pd.DataFrame:
    """Return the split table (docs/io_spec.md §5) of the *reference* dataset for one scenario and seed.

    ``obs`` holds cell metadata indexed by cell ID, with at least ``REQUIRED_OBS``.
    Only rows of ``scenario.reference`` are used; query-dataset rows are ignored.
    It never needs an expression matrix. ``query_classes`` is needed only for
    ``train_classes: shared`` in a cross-dataset scenario (the classes the query
    set will score).
    """
    ref = scenario.reference
    df = _metadata(obs, ref)
    label = df["label_harmonized"]

    if scenario.within:
        unknown = (set(scenario.eval_donors) | set(scenario.calibration_donors)) - set(df["donor"])
        if unknown:
            raise SplitError(f"donors {sorted(unknown)} are not in dataset {ref}")
        query_side = df["donor"].isin(scenario.eval_donors)
    else:
        unknown = set(scenario.calibration_donors) - set(df["donor"])
        if unknown:
            raise SplitError(f"calibration donors {sorted(unknown)} are not in reference {ref}")
        query_side = pd.Series(False, index=df.index)
    reference_side = ~query_side

    role = pd.Series(pd.NA, index=df.index, dtype="object")
    reason = pd.Series("", index=df.index, dtype="object")

    # 1. Cells that can never be supervised: author exclusions and QC failures.
    usable, unusable, combined = _unusable(df)
    challenge = (query_side & unusable & df["qc_pass"].astype(bool)
                 & df["exclusion_reason"].isin(rules.challenge_reasons))
    role[unusable] = "excluded"
    role[challenge] = "challenge_unlabeled"
    reason[unusable] = combined[unusable]

    # 2. Reference side: class removal, calibration donors, training candidates.
    ref_usable = usable & reference_side
    removed = ref_usable & label.isin(scenario.remove)
    role[removed], reason[removed] = "excluded", "class_removed:" + label[removed]
    if scenario.train_classes == "shared":
        if scenario.within:
            query_classes = set(label[usable & query_side])
        elif query_classes is None:
            raise SplitError("train_classes: shared in a cross-dataset scenario needs query_classes")
        not_shared = ref_usable & ~removed & ~label.isin(query_classes)
        role[not_shared], reason[not_shared] = "excluded", "class_not_in_query_task"
    else:
        not_shared = pd.Series(False, index=df.index)
    in_play = ref_usable & ~removed & ~not_shared
    calibration_donor = df["donor"].isin(scenario.calibration_donors)
    candidates = in_play & ~calibration_donor

    if rules.support_policy == "restrict":
        counts = label[candidates].value_counts()
        weak = set(counts[counts < rules.min_train_per_class].index)
        restricted = in_play & label.isin(weak)
        role[restricted], reason[restricted] = "excluded", "insufficient_train_support"
        in_play &= ~restricted
        candidates &= ~restricted

    train_classes = sorted(set(label[candidates]))
    calibration = in_play & calibration_donor & label.isin(train_classes)
    orphan = in_play & calibration_donor & ~label.isin(train_classes)
    role[calibration], reason[calibration] = "calibration", "calibration_donor"
    role[orphan], reason[orphan] = "excluded", "class_absent_from_train"

    # 3. Held-out donors (within-dataset only): seen classes are evaluation, unseen classes are open-set.
    query_usable = usable & query_side
    seen = query_usable & label.isin(train_classes)
    unseen = query_usable & ~label.isin(train_classes)
    role[seen], reason[seen] = "evaluation", "evaluation_donor"
    role[unseen] = "evaluation_open_set"
    reason[unseen] = np.where(label[unseen].isin(scenario.remove), "class_removed:" + label[unseen],
                              "class_not_in_training")

    # 4. The cap: per class, a seeded draw over sorted cell IDs.
    role[candidates], reason[candidates] = "train", "train_donor"
    if rules.cap_per_class is not None:
        for cls in train_classes:
            members = df.index[candidates & (label == cls)]
            if len(members) <= rules.cap_per_class:
                continue
            ordered = df.loc[members, "cell_id"].sort_values(kind="stable").index
            keep = seeded_rng(seed, "cap", cls).choice(len(ordered), size=rules.cap_per_class, replace=False)
            lost = ordered.delete(np.sort(keep))
            role[lost], reason[lost] = "train_reserve", f"above_cap_{rules.cap_per_class}"

    if role.isna().any():
        raise SplitError(f"{int(role.isna().sum())} cells received no role (bug in make_split)")
    df["role"], df["role_reason"] = role.astype(str), reason.astype(str)
    df["inner_fold"] = df["donor"].where(df["role"].isin(["train", "train_reserve"]), "")
    split = df[SPLIT_COLUMNS].sort_values(["dataset", "cell_id"], kind="stable").reset_index(drop=True)
    validate_split(split, scenario, rules)
    return split


def train_classes_of(split: pd.DataFrame) -> list[str]:
    """The classes the reference trains on: what a query cell must be to count as 'seen'."""
    return sorted(set(split.loc[split["role"] == "train", "label_harmonized"]))


def make_query_set(obs: pd.DataFrame, scenario: Scenario, train_classes: list[str],
                   rules: SplitRules) -> pd.DataFrame:
    """Which cells of the query dataset are scored, and how (cross-dataset scenarios only).

    No cell here is trained on, calibrated on or capped, and no seed is involved.
    ``train_classes`` comes from the reference split (``train_classes_of``); it is
    the same for every seed, because the cap never removes a class.
    """
    if scenario.within:
        raise SplitError("a within-dataset scenario keeps its held-out donor in the split; no query set")
    df = _metadata(obs, scenario.query)
    label = df["label_harmonized"]
    usable, unusable, combined = _unusable(df)
    role = pd.Series("excluded", index=df.index, dtype="object")
    reason = combined.copy()
    challenge = unusable & df["qc_pass"].astype(bool) & df["exclusion_reason"].isin(rules.challenge_reasons)
    role[challenge] = "challenge_unlabeled"
    seen = usable & label.isin(train_classes)
    unseen = usable & ~label.isin(train_classes)
    role[seen], reason[seen] = "evaluation", "query_dataset"
    role[unseen] = "evaluation_open_set"
    reason[unseen] = np.where(label[unseen].isin(scenario.remove), "class_removed:" + label[unseen],
                              "class_not_in_training")
    df["role"], df["role_reason"] = role.astype(str), reason.astype(str)
    query_set = df[QUERY_COLUMNS].sort_values(["dataset", "cell_id"], kind="stable").reset_index(drop=True)
    validate_query_set(query_set, scenario, train_classes)
    return query_set


def validate_query_set(query_set: pd.DataFrame, scenario: Scenario, train_classes: list[str]) -> None:
    problems = []
    if list(query_set.columns) != QUERY_COLUMNS:
        problems.append(f"columns are {list(query_set.columns)}, expected {QUERY_COLUMNS}")
    if set(query_set["dataset"]) != {scenario.query}:
        problems.append(f"query set holds datasets {sorted(set(query_set['dataset']))}, expected only {scenario.query}")
    bad = sorted(set(query_set["role"]) - set(QUERY_ROLES) - {"excluded"})
    if bad:
        problems.append(f"roles {bad} are not allowed in a query set")
    evaluated = set(query_set.loc[query_set["role"] == "evaluation", "label_harmonized"])
    open_set = set(query_set.loc[query_set["role"] == "evaluation_open_set", "label_harmonized"])
    if evaluated - set(train_classes):
        problems.append(f"evaluation classes absent from train: {sorted(evaluated - set(train_classes))}")
    if open_set & set(train_classes):
        problems.append(f"open-set classes present in train: {sorted(open_set & set(train_classes))}")
    for removed in scenario.remove:
        if removed in evaluated:
            problems.append(f"removed class '{removed}' is scored as seen 'evaluation'")
    if problems:
        raise SplitError("query set breaks the rules:\n  - " + "\n  - ".join(problems))


def validate_split(split: pd.DataFrame, scenario: Scenario, rules: SplitRules | None = None) -> None:
    """Re-check the hard rules on a finished split (also usable on a split read from disk)."""
    problems = []
    if set(split["dataset"]) - {scenario.reference}:
        problems.append(f"the split holds query-dataset cells: {sorted(set(split['dataset']) - {scenario.reference})}")
    if list(split.columns) != SPLIT_COLUMNS:
        problems.append(f"columns are {list(split.columns)}, expected {SPLIT_COLUMNS}")
    bad_roles = sorted(set(split["role"]) - set(ROLES))
    if bad_roles:
        problems.append(f"unknown roles {bad_roles}")
    if split.duplicated(["dataset", "cell_id"]).any():
        problems.append("a cell appears more than once")
    for dataset, part in split.groupby("dataset"):
        train_donors = set(part.loc[part["role"].isin(["train", "train_reserve"]), "donor"])
        calib_donors = set(part.loc[part["role"] == "calibration", "donor"])
        if train_donors & calib_donors:
            problems.append(f"{dataset}: donors {sorted(train_donors & calib_donors)} are in train and calibration")
        if scenario.reference == scenario.query:
            ref_donors = set(part.loc[part["role"].isin(REFERENCE_ROLES), "donor"])
            leaked = ref_donors & set(scenario.eval_donors)
            if leaked:
                problems.append(f"{dataset}: evaluation donors {sorted(leaked)} also hold reference cells")
    trained = set(split.loc[split["role"] == "train", "label_harmonized"])
    for removed in scenario.remove:
        in_reference = split["role"].isin(REFERENCE_ROLES) & (split["label_harmonized"] == removed)
        if in_reference.any():
            problems.append(f"removed class '{removed}' still has {int(in_reference.sum())} reference cells")
        wrongly_scored = (split["role"] == "evaluation") & (split["label_harmonized"] == removed)
        if wrongly_scored.any():
            problems.append(f"removed class '{removed}' has cells scored as seen 'evaluation'")
    evaluated = set(split.loc[split["role"] == "evaluation", "label_harmonized"])
    calibrated = set(split.loc[split["role"] == "calibration", "label_harmonized"])
    open_set = set(split.loc[split["role"] == "evaluation_open_set", "label_harmonized"])
    if evaluated - trained:
        problems.append(f"evaluation classes absent from train: {sorted(evaluated - trained)}")
    if calibrated - trained:
        problems.append(f"calibration classes absent from train: {sorted(calibrated - trained)}")
    if open_set & trained:
        problems.append(f"open-set classes present in train: {sorted(open_set & trained)}")
    if rules is not None and rules.cap_per_class is not None:
        per_class = split.loc[split["role"] == "train", "label_harmonized"].value_counts()
        if (per_class > rules.cap_per_class).any():
            problems.append(f"classes above the cap: {per_class[per_class > rules.cap_per_class].to_dict()}")
    if problems:
        raise SplitError("split breaks the rules:\n  - " + "\n  - ".join(problems))


def split_counts(split: pd.DataFrame) -> pd.DataFrame:
    """Cells per role x dataset x donor x class: the table every report starts from."""
    return (
        split.groupby(["role", "dataset", "donor", "label_harmonized"], observed=True)
        .size().rename("n_cells").reset_index()
        .sort_values(["role", "dataset", "donor", "label_harmonized"]).reset_index(drop=True)
    )


def support_restrictions(split: pd.DataFrame, rules: SplitRules, query_set: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every documented restriction (proposal IV.E: report, never silently resample).

    Evaluation and open-set support come from the held-out donors in the split
    (within-dataset) or from the query set (cross-dataset).
    """
    rows = []
    trained = train_classes_of(split)
    scored = split if query_set is None else pd.concat([split[QUERY_COLUMNS], query_set], ignore_index=True)

    def per_class(role: str) -> pd.Series:
        table = scored if role in ("evaluation", "evaluation_open_set") else split
        return table.loc[table["role"] == role, "label_harmonized"].value_counts()

    train_n, calib_n, eval_n = per_class("train"), per_class("calibration"), per_class("evaluation")
    for cls in trained:
        if calib_n.get(cls, 0) < rules.min_calibration_per_class:
            rows.append({"class": cls, "check": "calibration_support", "observed": int(calib_n.get(cls, 0)),
                         "threshold": rules.min_calibration_per_class,
                         "consequence": "no class-conditional calibration; pooled calibration only (VII.C)"})
        if train_n.get(cls, 0) < rules.min_train_per_class:
            rows.append({"class": cls, "check": "train_support", "observed": int(train_n.get(cls, 0)),
                         "threshold": rules.min_train_per_class,
                         "consequence": "trained on few cells; interpret per-class results with care"})
    for cls in sorted(set(scored.loc[scored["role"] == "evaluation", "label_harmonized"])):
        if eval_n.get(cls, 0) < rules.min_eval_per_class:
            rows.append({"class": cls, "check": "evaluation_support", "observed": int(eval_n.get(cls, 0)),
                         "threshold": rules.min_eval_per_class,
                         "consequence": "too few evaluation cells to score this class reliably"})
    train = split[split["role"] == "train"]
    for cls, part in train.groupby("label_harmonized"):
        share = part["donor"].value_counts(normalize=True)
        if share.iloc[0] >= rules.donor_dominance:
            rows.append({"class": cls, "check": "donor_dominance", "observed": round(float(share.iloc[0]), 3),
                         "threshold": rules.donor_dominance,
                         "consequence": f"{share.index[0]} supplies most training cells; donor and class are confounded"})
    for cls in sorted(set(scored.loc[scored["role"] == "evaluation_open_set", "label_harmonized"])):
        n = int((scored.loc[scored["role"] == "evaluation_open_set", "label_harmonized"] == cls).sum())
        rows.append({"class": cls, "check": "open_set_query_class", "observed": n, "threshold": None,
                     "consequence": "scored in the separate open-set analysis, not closed-set metrics (VI.B)"})
    return pd.DataFrame(rows, columns=["class", "check", "observed", "threshold", "consequence"])


def split_path(folder: str | Path, version: str, split_id: str) -> Path:
    return Path(folder) / version / f"{split_id}.tsv.gz"


def write_split(split: pd.DataFrame, rules: SplitRules, folder: str | Path, version: str, split_id: str,
                query_set: pd.DataFrame | None = None) -> Path:
    """Write the split, its count table, its restrictions, and update the manifest."""
    path = split_path(folder, version, split_id)
    write_tsv(split, path)
    write_tsv(split_counts(split), path.with_name(f"{split_id}.counts.tsv"))
    write_tsv(support_restrictions(split, rules, query_set), path.with_name(f"{split_id}.restrictions.tsv"))
    update_manifest(path.parent / "manifest.tsv", split_id, path, len(split))
    return path


def query_set_path(folder: str | Path, version: str, scenario_id: str) -> Path:
    return Path(folder) / version / f"{scenario_id}.tsv.gz"


def write_query_set(query_set: pd.DataFrame, folder: str | Path, version: str, scenario_id: str) -> Path:
    path = query_set_path(folder, version, scenario_id)
    write_tsv(query_set, path)
    counts = (query_set.groupby(["role", "donor", "label_harmonized"], observed=True).size()
              .rename("n_cells").reset_index().sort_values(["role", "donor", "label_harmonized"]))
    write_tsv(counts.reset_index(drop=True), path.with_name(f"{scenario_id}.counts.tsv"))
    update_manifest(path.parent / "manifest.tsv", scenario_id, path, len(query_set), key="query_set_id")
    return path


def read_query_set(path: str | Path) -> pd.DataFrame:
    table = read_tsv(path)
    if list(table.columns) != QUERY_COLUMNS:
        raise SplitError(f"{path}: columns {list(table.columns)} differ from {QUERY_COLUMNS}")
    return table


def update_manifest(manifest: Path, split_id: str, path: Path, n_cells: int, key: str = "split_id") -> None:
    """One row per split (or query set) file with its checksum (proposal IV.D, io_spec §8 last row)."""
    row = pd.DataFrame([{key: split_id, "file": path.name, "n_cells": str(n_cells), "sha256": sha256_file(path)}])
    if manifest.exists():
        table = read_tsv(manifest)
        table = pd.concat([table[table[key] != split_id], row], ignore_index=True)
    else:
        table = row
    write_tsv(table.sort_values(key).reset_index(drop=True), manifest)


def read_split(path: str | Path) -> pd.DataFrame:
    split = read_tsv(path)
    if list(split.columns) != SPLIT_COLUMNS:
        raise SplitError(f"{path}: columns {list(split.columns)} differ from {SPLIT_COLUMNS}")
    return split
