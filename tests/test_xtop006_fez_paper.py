"""Verification tests for `OSIRIS-XTOP-006B-FEZ` (`dc23ouqnj2vs73dg3bog`) and
`papers/xtop006_non_clifford_kraus_fez/` macro-locked REVTeX manuscript package.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

from osiris_cli import claims

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "papers" / "xtop006_non_clifford_kraus_fez"
XB_DIR = REPO_ROOT / "experiments" / "xtop006b_fez"


def _load_gen():
    p = PAPER_DIR / "generate_xtop006_paper_assets.py"
    spec = importlib.util.spec_from_file_location("gen_xtop006_paper", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_xtop006b_fez_hardware_attested_bundle() -> None:
    man = json.loads((XB_DIR / "xtop006b_fez_staging_manifest.json").read_text(encoding="utf-8"))
    res = json.loads((XB_DIR / "physical_xtop006b_fez_results.json").read_text(encoding="utf-8"))
    qpy_sha = hashlib.sha256((XB_DIR / "xtop006b_fez_staged_circuits.qpy").read_bytes()).hexdigest()
    gz_sha = hashlib.sha256((XB_DIR / "physical_xtop006b_fez_raw_counts.json.gz").read_bytes()).hexdigest()

    assert qpy_sha == man["qpy_sha256"] == res["qpy_sha256"]
    assert gz_sha == res["raw_counts_gz_sha256"]
    assert res["execution_tier"] == "HARDWARE_ATTESTED"
    assert res["provider_job_id"] == "dc23ouqnj2vs73dg3bog"
    assert res["hypotheses"]["H2_confirmatory_intra_block_4twirl_discriminator"]["verdict"] == "SUPPORTED"
    for row in res["intra_block_4twirl_contrasts"]:
        assert row["p_holm_adjusted"] < 0.01
        assert row["paired_t_stat"] > 10.0


def test_xtop006_paper_deterministic_and_macro_clean() -> None:
    mod = _load_gen()
    mod.generate_all()
    snap1 = (PAPER_DIR / "SHA256SUMS.txt").read_text(encoding="utf-8")
    mod.generate_all()
    snap2 = (PAPER_DIR / "SHA256SUMS.txt").read_text(encoding="utf-8")
    assert snap1 == snap2

    gen_tex = (PAPER_DIR / "generated.tex").read_text(encoding="utf-8")
    tex = (PAPER_DIR / "xtop006_non_clifford_kraus_fez.tex").read_text(encoding="utf-8")
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", gen_tex))
    used_xtop = set(re.findall(r"\\(Xtop[A-Za-z]+)", tex))
    assert used_xtop <= defined, f"Undefined macros: {used_xtop - defined}"
    assert tex.count("{") == tex.count("}")
    assert claims.flagged_in_reply(tex, []) == []

    cap = json.loads((PAPER_DIR / "reproducibility_capsule.json").read_text(encoding="utf-8"))
    assert cap["execution_tier"] == "HARDWARE_ATTESTED"
    assert cap["provider_job_ids"] == ["dc1tqaanj2vs73dfrt40", "dc23ouqnj2vs73dg3bog"]
