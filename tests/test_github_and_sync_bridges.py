"""
Unit Tests for OSIRIS GitHub Bridge and Dual-Node Dev Sync Subsystems
====================================================================
Verifies:
  - GitHub PR inspection and repo management (Read-only tier)
  - Strict SYS-021 blocking on un-granted mutating GitHub directives
  - Dual-Node Substrate detection (Node Alpha vs Node Beta)
  - Bluetooth PAN and Wireless ADB sync runbook formatting
"""

import platform
import unittest
from unittest.mock import patch

from osiris_cli.github_bridge import GitHubBridge
from osiris_cli.sync_bridge import DualNodeSyncBridge


class TestGitHubBridge(unittest.TestCase):
    """Test GitHub CLI integration and governance tiers."""

    def setUp(self):
        self.bridge = GitHubBridge()

    def test_auth_status_inspection(self):
        """Verify GitHub auth check parses status without crashing."""
        auth = self.bridge.check_auth()
        self.assertIn("installed", auth)
        self.assertIn("authenticated", auth)
        if auth["installed"] and auth["authenticated"]:
            # Whoever is logged in to gh here; the account changed on 2026-11-09 and may again.
            self.assertTrue(auth.get("account"))
            self.assertIn("repo", auth.get("scopes", []))

    def test_read_only_pr_listing(self):
        """Verify list_prs returns structured response."""
        res = self.bridge.list_prs(limit=5)
        self.assertIn(res["status"], ("SUCCESS", "ERROR"))
        if res["status"] == "SUCCESS":
            self.assertIsInstance(res.get("prs"), list)

    def test_read_only_pr_view(self):
        """Verify view_pr extracts details for existing PR."""
        res = self.bridge.view_pr(27)
        self.assertIn(res["status"], ("SUCCESS", "ERROR"))
        if res["status"] == "SUCCESS":
            pr = res.get("pr", {})
            self.assertEqual(pr.get("number"), 27)
            self.assertEqual(pr.get("headRefName"), "feat/gemma-substrate-governed-agents")

    def test_mutating_merge_fails_closed_without_grant(self):
        """Verify SYS-021: merge without payload-bound grant is strictly blocked."""
        res = self.bridge.merge_pr(27)
        self.assertEqual(res["status"], "BLOCKED")
        self.assertIn("SYS-021", res["reason"])

    def test_dashboard_formatting(self):
        """Verify format_dashboard outputs governance card (gh stubbed: same on any machine)."""
        auth = {"installed": True, "authenticated": True, "account": "octo", "scopes": ["repo"]}
        prs = {"status": "SUCCESS", "prs": [{"number": 7, "title": "t", "headRefName": "b"}]}
        with patch.object(self.bridge, "check_auth", return_value=auth), \
                patch.object(self.bridge, "list_prs", return_value=prs):
            dash = self.bridge.format_dashboard()
        self.assertIn("OSIRIS::GITHUB REPOSITORY & PR SUBSYSTEM", dash)
        self.assertIn("Governance Tier", dash)
        self.assertIn("octo", dash)
        self.assertIn("#7", dash)

    def test_dashboard_without_gh_auth_or_binary(self):
        """An unauthenticated or missing gh shows the remedy, never a governance tier."""
        for auth, hint in (({"installed": True, "authenticated": False}, "gh auth login"),
                           ({"installed": False, "authenticated": False}, "not installed")):
            with patch.object(self.bridge, "check_auth", return_value=auth):
                dash = self.bridge.format_dashboard()
            self.assertIn(hint, dash)
            self.assertNotIn("Governance Tier", dash)


class TestDualNodeSyncBridge(unittest.TestCase):
    """Test Dual-Node Dev Sync diagnostics and runbook."""

    def test_substrate_detection(self):
        """Verify node substrate detection accurately labels hardware."""
        sub = DualNodeSyncBridge.detect_substrate()
        self.assertIn("node_id", sub)
        self.assertIn("machine", sub)
        self.assertIn("is_arm64", sub)
        # The label must agree with the machine the test runs on (phone or workstation)
        is_arm64 = platform.machine().lower() in ("aarch64", "arm64")
        self.assertEqual(sub["is_arm64"], is_arm64)
        self.assertIn("Node Alpha" if is_arm64 else "Node Beta", sub["node_id"])

    def test_ssh_listener_probe(self):
        """Verify non-privileged port probe runs without raising exceptions."""
        is_listening = DualNodeSyncBridge.check_ssh_listener(port=8022, timeout=0.1)
        self.assertIsInstance(is_listening, bool)

    def test_runbook_contains_core_axioms_and_channels(self):
        """Verify runbook documents Bluetooth PAN, Wireless Debugging, and Axiom."""
        runbook = DualNodeSyncBridge.format_runbook()
        self.assertIn("Cross-node transfer is a data operation, not an authorization operation.", runbook)
        self.assertIn("Android Wireless Debugging", runbook)
        self.assertIn("Bluetooth PAN", runbook)
        self.assertIn("8022", runbook)
        self.assertIn("Scoped Storage", runbook)

    def test_status_card_formatting(self):
        """Verify status card formatting."""
        card = DualNodeSyncBridge.format_status_card()
        self.assertIn("OSIRIS::DUAL-NODE SYNC & DEVICE BRIDGE", card)
        self.assertIn("Detected Substrate", card)
        self.assertIn("SSH Daemon", card)


if __name__ == "__main__":
    unittest.main()
