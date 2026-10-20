"""Test trivial -- prouve que la chaîne lint/format/tests/CI fonctionne avant tout code réel."""

import multi_agent_research_team


def test_le_package_s_importe():
    assert multi_agent_research_team is not None
