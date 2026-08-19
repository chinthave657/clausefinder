"""Clause-set alignment tests — synthetic fixtures only, no index, no API keys.

Covers unchanged / modified / added / removed / renumbered cases with
realistic TS 38.331-style clause numbering, per design §3 (diff agent) and
the P0 constraint that the dev index carries Rel-18 only.
"""
from agent.tools.clause_align import (
    TITLE_SIM_THRESHOLD,
    ClauseSide,
    _title_sim,
    align_clauses,
    render_diff,
)


def _side(tag, release, clause, title, text):
    return ClauseSide(tag=tag, release=release, clause=clause, title=title,
                      text=text, chunk={"spec": "TS 38.331", "version": "18.2.0"})


# --- fixtures ---------------------------------------------------------------
# Rel-17: 3 clauses. Rel-18: same clause identical, same clause modified,
# one clause removed (5.3.5.9), one clause added (5.3.5.20 LTM, new in
# Rel-18), one clause renumbered (5.3.5.11 -> 5.3.5.11a, title preserved).

R17 = [
    _side("[C1]", "Rel-17", "5.3.5.3", "Reception of an RRCReconfiguration by the UE",
          "1> the UE shall perform the reconfiguration in accordance with the received "
          "RRCReconfiguration message.\n2> the UE shall submit the RRCReconfigurationComplete "
          "message to lower layers for transmission."),
    _side("[C2]", "Rel-17", "5.3.5.9", "Void",
          "This clause is intentionally left void in Rel-17 and removed in Rel-18."),
    _side("[C3]", "Rel-17", "5.3.5.11", "Full configuration",
          "The UE shall perform the full configuration procedure upon reception of fullConfig."),
]

R18 = [
    _side("[C4]", "Rel-18", "5.3.5.3", "Reception of an RRCReconfiguration by the UE",
          "1> the UE shall perform the reconfiguration in accordance with the received "
          "RRCReconfiguration message.\n2> the UE shall submit the RRCReconfigurationComplete "
          "message to lower layers for transmission."),
    _side("[C5]", "Rel-18", "5.3.5.20", "L1/L2 triggered mobility (LTM) execution",
          "The UE shall execute the LTM cell switch as configured by the candidate "
          "configuration selected via the LTM command MAC CE."),
    _side("[C6]", "Rel-18", "5.3.5.11a", "Full configuration",
          "The UE shall perform the full configuration procedure upon reception of fullConfig, "
          "including release of SCells and secondary cell groups."),
]


def test_unchanged_clause_matched_by_number():
    pairs = align_clauses(R17[:1], R18[:1])
    assert len(pairs) == 1
    assert pairs[0].kind == "unchanged"
    assert pairs[0].a.clause == "5.3.5.3" and pairs[0].b.clause == "5.3.5.3"


def test_removed_clause_has_no_b_side():
    pairs = align_clauses(R17, R18)
    removed = [p for p in pairs if p.kind == "removed"]
    assert len(removed) == 1
    assert removed[0].a.clause == "5.3.5.9"
    assert removed[0].b is None


def test_added_clause_has_no_a_side():
    pairs = align_clauses(R17, R18)
    added = [p for p in pairs if p.kind == "added"]
    assert len(added) == 1
    assert added[0].b.clause == "5.3.5.20"
    assert added[0].a is None


def test_renumbered_clause_matched_by_title_and_flagged():
    pairs = align_clauses(R17, R18)
    modified = [p for p in pairs if p.kind == "modified"]
    renum = [p for p in modified if p.renumbered]
    assert len(renum) == 1
    assert renum[0].a.clause == "5.3.5.11" and renum[0].b.clause == "5.3.5.11a"
    assert renum[0].diff_lines  # text differs (release note appended) -> real diff lines


def test_full_release_alignment_partitions_all_clauses():
    pairs = align_clauses(R17, R18)
    kinds = sorted(p.kind for p in pairs)
    # 1 unchanged (5.3.5.3), 1 renumbered-modified (5.3.5.11->11a), 1 removed
    # (5.3.5.9), 1 added (5.3.5.20)
    assert kinds == ["added", "modified", "removed", "unchanged"]


def test_title_sim_below_threshold_falls_back_to_removed_added():
    a = [_side("[C1]", "Rel-17", "9.1.1", "Measurement configuration",
               "old text about measurement gaps")]
    b = [_side("[C2]", "Rel-18", "9.1.1b", "Random access procedure",
               "completely unrelated text about PRACH preambles")]
    pairs = align_clauses(a, b)
    assert sorted(p.kind for p in pairs) == ["added", "removed"]
    assert _title_sim(a[0].title, b[0].title) < TITLE_SIM_THRESHOLD


def test_title_sim_uses_token_set_ratio_ignores_word_order_and_extras():
    # token_set_ratio should score this pair well above threshold even though
    # word order differs and side b has an extra qualifying word.
    score = _title_sim("Cell group configuration",
                       "Configuration of the cell group (NR)")
    assert score >= TITLE_SIM_THRESHOLD


def test_render_diff_omits_unchanged_and_labels_each_kind():
    pairs = align_clauses(R17, R18)
    rendered = render_diff(pairs, "Rel-17", "Rel-18")
    assert "5.3.5.3" not in rendered  # unchanged clause not rendered
    assert "== ADDED in Rel-18: §5.3.5.20" in rendered
    assert "== REMOVED since Rel-17: §5.3.5.9" in rendered
    assert "== MODIFIED: §5.3.5.11" in rendered
    assert "[renumbered §5.3.5.11 -> §5.3.5.11a]" in rendered


def test_render_diff_empty_when_all_unchanged():
    pairs = align_clauses(R17[:1], R18[:1])
    assert render_diff(pairs, "Rel-17", "Rel-18") == ""


def test_align_clauses_symmetric_empty_sides():
    assert align_clauses([], []) == []
    only_a = align_clauses(R17[:1], [])
    assert only_a[0].kind == "removed"
    only_b = align_clauses([], R18[:1])
    assert only_b[0].kind == "added"


def test_modified_diff_lines_capped_and_normalized():
    long_a = _side("[C1]", "Rel-17", "6.1.1", "Long clause",
                   "\n".join(f"line {i} shall do something" for i in range(60)))
    long_b = _side("[C2]", "Rel-18", "6.1.1", "Long clause",
                   "\n".join(f"line {i} shall do something else" for i in range(60)))
    pairs = align_clauses([long_a], [long_b])
    assert pairs[0].kind == "modified"
    assert len(pairs[0].diff_lines) <= 41  # MAX_DIFF_LINES + truncation marker
    assert pairs[0].diff_lines[-1] == "… (diff truncated)"


def test_markdown_noise_normalized_to_unchanged():
    # citations._norm strips markdown emphasis/escapes and smart punctuation;
    # two textually-equivalent renderings (same bullet marker) must align as
    # unchanged even though their raw markdown/quoting differs.
    a = _side("[C1]", "Rel-17", "5.3.5.3", "Reception of an RRCReconfiguration",
              "1> the UE **shall** submit the “RRCReconfigurationComplete” message.")
    b = _side("[C2]", "Rel-18", "5.3.5.3", "Reception of an RRCReconfiguration",
              "1> the UE shall submit the \"RRCReconfigurationComplete\" message.")
    pairs = align_clauses([a], [b])
    assert pairs[0].kind == "unchanged"
