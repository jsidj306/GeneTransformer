"""Unit tests for Enrichr result parsing (pure functions, no network)."""
from src.enrichment import parse_rows, short_term


def test_parse_rows_basic():
    # Enrichr row = [rank, term, p, odds_ratio, combined, genes(list), adj_p, old_p, old_adj_p]
    row = [1, "Pathways in cancer", 1e-29, 722.44, 47893.85,
           ["RB1", "SMAD4", "TP53"], 2.44e-27, 0.0, 0.0]
    parsed = parse_rows("KEGG_2021_Human", [row])
    assert len(parsed) == 1
    r = parsed[0]
    assert r["library"] == "KEGG_2021_Human"
    assert r["term"] == "Pathways in cancer"
    assert r["overlap"] == 3
    assert r["odds_ratio"] == 722.44
    assert r["adj_p_value"] == 2.44e-27
    assert r["genes"] == ["RB1", "SMAD4", "TP53"]


def test_parse_rows_multiple_overlaps():
    rows = [
        [1, "A (GO:0001)", 1e-3, 2.0, 10.0, ["G1"], 0.01, 0, 0],
        [2, "B", 1e-2, 3.0, 5.0, ["G1", "G2"], 0.5, 0, 0],
    ]
    parsed = parse_rows("GO_Biological_Process_2023", rows)
    assert [r["overlap"] for r in parsed] == [1, 2]
    assert parsed[0]["library"] == "GO_Biological_Process_2023"


def test_short_term():
    assert short_term("Muscle Contraction (GO:0006936)") == "Muscle Contraction"
    assert short_term("Assembly of Collagen Fibrils R-HSA-2022090") == "Assembly of Collagen Fibrils"
    assert short_term("DNA Repair (Homo sapiens)") == "DNA Repair"
    assert short_term("No Suffix Here") == "No Suffix Here"
