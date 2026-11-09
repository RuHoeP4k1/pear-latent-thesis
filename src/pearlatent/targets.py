"""The PLS-DA targets of charter rule 6 beside `binary_23` (notebook 14; Ruben, 9 November 2026).

Rule 6 asks for `binary_123` beside `binary_23`, a severity score fitted on grade 0 against 3
only, and two curves: any defect, and cavity-free browning. Each target says which fruit it uses,
which grade describes them, where the 0/1 label cuts, and which fruit a model may be fitted on.

Severity score: the model sees only the extreme grades, so grades 1 and 2 play no part in the
fit. It still predicts every held-out fruit. Where grades 1 and 2 land on that score shows whether
they lie between the extremes, without the model having been told about them.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class Target:
    """One target. `positive_from`: label 1 from this grade up. `fit_grades`: None = all grades."""

    label: str
    grade_column: str
    cavity_free: bool
    positive_from: int
    fit_grades: tuple[int, ...] | None = None


TARGETS = {
    "binary_123": Target(
        "any defect: defective grade ≥ 1, all fruit", "defective", False, 1
    ),
    "browning_2plus_cavity_free": Target(
        "browning grade ≥ 2, cavity-free fruit", "browning", True, 2
    ),
    "severity_defective": Target(
        "severity: defective grade 0 against 3, all fruit",
        "defective",
        False,
        3,
        (0, 3),
    ),
    "severity_browning_cavity_free": Target(
        "severity: browning grade 0 against 3, cavity-free fruit",
        "browning",
        True,
        3,
        (0, 3),
    ),
}


def target_table(fruit: pl.DataFrame, name: str) -> pl.DataFrame:
    """The fruit of one target, in the order of `fruit`, with `grade`, `y` and `fit_on`.

    In: a table with `browning`, `cavity`, `defective`; the target name. Out: the kept rows plus
    `grade` (the grade this target is about), `y` (0/1 label) and `fit_on` (may a model be fitted
    on this fruit). For a severity target, `y` of grades 1 and 2 is 0 but is never used: they are
    not fitted on and not scored by the AUC of the target. Prints how many fruit went in and out.
    """
    if name not in TARGETS:
        raise ValueError(f"unknown target {name!r}; known: {sorted(TARGETS)}")
    target = TARGETS[name]
    kept = fruit.filter(pl.col("cavity") == 0) if target.cavity_free else fruit
    print(f"target {name}: {fruit.height} fruit in, {kept.height} out")
    grade = pl.col(target.grade_column)
    fit_on = (
        pl.lit(True) if target.fit_grades is None else grade.is_in(target.fit_grades)
    )
    return kept.with_columns(
        grade=grade,
        y=(grade >= target.positive_from).cast(pl.Int8),
        fit_on=fit_on,
    )
