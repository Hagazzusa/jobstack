"""Run with python3 test/test-packet.py. All data is synthetic."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


TEST_DIR = Path(__file__).resolve().parent
CLI = TEST_DIR.parent / "bin" / "jobstack-packet"
JOB_ID = "test-job"


class PacketTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".packet-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = self.root / "state"
        self.source = self.root / "final"
        self.answer = self.source / "answers" / "question.txt"
        self.attachment = self.source / "attachments" / "nested" / "portfolio.pdf"
        self.evidence = self.root / "confirmation.txt"
        self.failure_evidence = self.root / "rejection.txt"
        self.answer.parent.mkdir(parents=True)
        self.attachment.parent.mkdir(parents=True)
        self.answer.write_text("Final answer.\n")
        self.attachment.write_bytes(b"Test attachment\x00\xff")
        self.evidence.write_text("Test portal: submission confirmed.\n")
        self.failure_evidence.write_text("Test portal: attempt 1 rejected because the attachment is invalid. Nothing was submitted.\n")
        self.identity = {"id": JOB_ID, "company": "Example", "position": "Engineer"}
        (self.source / "job.json").write_text(json.dumps(self.identity))

    def run_cli(self, command, *args, job_id=JOB_ID, store=None):
        return subprocess.run(
            [str(CLI), "--state-dir", str(store or self.store), command, job_id, *map(str, args)],
            capture_output=True, text=True, check=False,
        )

    def ok(self, command, *args, **kwargs):
        result = self.run_cli(command, *args, **kwargs)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        return json.loads(result.stdout)

    def rejected(self, message, command, *args, **kwargs):
        result = self.run_cli(command, *args, **kwargs)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn(message, result.stderr)

    def snapshot(self):
        return self.ok("snapshot", self.source)

    def submit_receipt(self, revision, *args):
        return self.ok("receipt", revision, "--success", "--confirmation", "Example-100", "--evidence", self.evidence, *args)

    def failure_args(self, attempt=1, reason="Invalid attachment", evidence=None):
        return "--attempt", attempt, "--not-submitted", "--reason", reason, "--evidence", evidence or self.failure_evidence

    def test_exact_approval_and_stable_submitted_evidence(self):
        self.assertEqual(self.ok("status"), {"job_id": JOB_ID, "status": "missing"})
        initial = self.snapshot()
        revision = initial["revision"]
        archive = Path(initial["snapshot_path"])
        self.assertEqual(initial["status"], "snapshotted")
        self.assertEqual(initial["job"], self.identity)
        self.assertNotIn(self.source, archive.parents)
        self.assertEqual((archive / self.attachment.relative_to(self.source)).read_bytes(), self.attachment.read_bytes())
        self.rejected("Approve the exact revision", "ready", revision)
        self.assertEqual(self.ok("approve", revision)["status"], "approved")
        self.rejected("Use ready", "receipt", revision, "--success", "--confirmation", "success", "--evidence", self.evidence)
        pending = self.ok("ready", revision)
        self.assertEqual(pending["status"], "pending")
        self.assertEqual(pending["snapshot_path"], str(archive))

        # Editing the working draft after handoff leaves the authorized archive available for reconciliation.
        self.answer.write_text("Edited draft.\n")
        expected = self.evidence.read_bytes()
        submitted = self.submit_receipt(revision)
        self.assertEqual(submitted["status"], "submitted")
        self.assertEqual(submitted["receipt"]["revision"], revision)
        self.assertEqual(submitted["receipt"]["confirmation"], "Example-100")
        self.assertEqual(submitted["receipt"]["filename"], "confirmation.txt")
        shutil.rmtree(self.source)
        self.evidence.unlink()
        self.assertEqual(self.ok("status"), submitted)
        self.assertEqual((archive / "answers" / "question.txt").read_text(), "Final answer.\n")
        self.assertEqual(Path(submitted["receipt"]["evidence_path"]).read_bytes(), expected)
        self.rejected("Do not retry", "ready", revision)
        self.rejected("already recorded", "receipt", revision, "--success", "--confirmation", "duplicate", "--evidence", self.evidence)

    def test_changed_answers_attachments_and_identity_invalidate_approval(self):
        first = self.snapshot()
        self.ok("approve", first["revision"])
        self.rejected("already approved", "approve", first["revision"])
        self.rejected("already the current revision", "snapshot", self.source)
        self.answer.rename(self.answer.with_name("renamed.txt"))
        self.assertEqual(self.ok("status")["status"], "changed")
        self.rejected("input changed", "ready", first["revision"])
        second = self.snapshot()
        self.assertNotEqual(second["revision"], first["revision"])
        self.rejected("no longer current", "approve", first["revision"])
        self.rejected("Approve the exact revision", "ready", second["revision"])
        self.attachment.write_bytes(b"Modified attachment")
        self.rejected("input changed", "approve", second["revision"])
        third = self.snapshot()
        (self.source / "job.json").write_text(json.dumps({**self.identity, "position": "Analyst"}))
        self.assertEqual(self.ok("status")["status"], "changed")
        fourth = self.snapshot()
        self.assertNotEqual(fourth["revision"], third["revision"])
        self.assertEqual(fourth["job"]["position"], "Analyst")

    def test_pending_claim_is_exclusive_and_invalid_evidence_does_not_settle_it(self):
        revision = self.snapshot()["revision"]
        self.ok("approve", revision)
        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = list(pool.map(lambda _: self.run_cli("ready", revision), range(2)))
        self.assertEqual(sorted(attempt.returncode for attempt in attempts), [0, 1])
        self.assertIn("Do not retry", next(attempt.stderr for attempt in attempts if attempt.returncode))
        self.rejected("already pending", "snapshot", self.source)
        self.rejected("no longer current", "receipt", "0" * 64, "--success", "--confirmation", "success", "--evidence", self.evidence)
        self.rejected("--success", "receipt", revision, "--confirmation", "success", "--evidence", self.evidence)
        self.rejected("nonempty --confirmation", "receipt", revision, "--success", "--confirmation", " ", "--evidence", self.evidence)
        self.rejected("outside the packet", "receipt", revision, "--success", "--confirmation", "success", "--evidence", self.answer)
        empty = self.root / "empty.txt"
        empty.touch()
        self.rejected("must be nonempty", "receipt", revision, "--success", "--confirmation", "success", "--evidence", empty)
        self.assertEqual(self.ok("status")["status"], "pending")
        self.assertEqual(self.submit_receipt(revision)["status"], "submitted")

    def test_invalid_layout_and_altered_archive_are_rejected(self):
        self.rejected("separate directories", "snapshot", self.source, store=self.source / "state")
        self.assertFalse((self.source / "state").exists())
        self.rejected("id must match", "snapshot", self.source, job_id="other-job")
        self.rejected("JOB_ID", "status", job_id="../escape")
        foreign = self.source / "notes.txt"
        foreign.touch()
        self.rejected("Packet root requires", "snapshot", self.source)
        foreign.unlink()
        self.answer.unlink()
        self.rejected("nonempty final answer", "snapshot", self.source)
        self.answer.symlink_to(self.evidence)
        self.rejected("jobstack-packet:", "snapshot", self.source)
        self.answer.unlink()
        self.answer.write_text("Restored.\n")
        snapshot = self.snapshot()
        archive = Path(snapshot["snapshot_path"])
        self.assertEqual(archive.stat().st_mode & 0o222, 0)
        (archive / "answers" / "question.txt").chmod(0o600)
        (archive / "answers" / "question.txt").write_text("Tampered.\n")
        self.rejected("integrity verification", "approve", snapshot["revision"])

    def test_desktop_metadata_does_not_invalidate_pending_revision(self):
        (self.source / ".DS_Store").touch()
        initial = self.snapshot()
        revision = initial["revision"]
        archive = Path(initial["snapshot_path"])
        self.assertFalse((archive / ".DS_Store").exists())
        (self.answer.parent / ".DS_Store").touch()
        self.ok("approve", revision)
        self.ok("ready", revision)
        # Simulate Finder metadata in an older writable archive.
        (archive / "answers").chmod(0o700)
        (archive / "answers" / ".DS_Store").touch()
        self.assertEqual(self.submit_receipt(revision)["status"], "submitted")

    def test_checkout_cannot_hold_private_files(self):
        self.rejected("outside the JobStack checkout", "snapshot", TEST_DIR)
        self.rejected("outside the JobStack checkout", "snapshot", self.source, store=TEST_DIR)
        revision = self.snapshot()["revision"]
        self.ok("approve", revision)
        self.ok("ready", revision)
        self.rejected("outside the JobStack checkout", "receipt", revision, "--success",
                      "--confirmation", "Example", "--evidence", Path(__file__).resolve())

    def test_altered_archived_receipt_is_rejected(self):
        revision = self.snapshot()["revision"]
        self.ok("approve", revision)
        self.ok("ready", revision)
        receipt = self.submit_receipt(revision)["receipt"]
        Path(receipt["evidence_path"]).write_text("Replaced receipt data.\n")
        self.rejected("Stored success evidence failed integrity verification", "status")

    def test_failed_requires_a_pending_revision_and_confirmed_non_submission(self):
        revision = self.snapshot()["revision"]
        self.ok("approve", revision)
        self.rejected("No pending attempt", "failed", revision, *self.failure_args())
        pending = self.ok("ready", revision)
        self.assertEqual(pending["attempt"], 1)
        self.rejected("no longer current", "failed", "0" * 64, *self.failure_args())
        self.rejected("current pending attempt", "failed", revision, *self.failure_args(attempt=2))
        self.rejected("--not-submitted", "failed", revision, "--attempt", 1,
                      "--reason", "Connection lost", "--evidence", self.failure_evidence)
        self.rejected("nonempty --reason", "failed", revision, *self.failure_args(reason=" "))
        empty = self.root / "empty-rejection.txt"
        empty.touch()
        self.rejected("must be nonempty", "failed", revision, *self.failure_args(evidence=empty))
        self.assertEqual(self.ok("status"), pending)
        self.rejected("Do not retry", "ready", revision)

    def test_failed_archives_evidence_and_requires_fresh_approval_on_same_revision(self):
        revision = self.snapshot()["revision"]
        self.ok("approve", revision)
        pending = self.ok("ready", revision)
        reconciled = self.ok("failed", revision, *self.failure_args())
        self.assertEqual(reconciled["status"], "snapshotted")
        self.assertIsNone(reconciled["approved_at"])
        self.assertIsNone(reconciled["attempt"])
        self.assertIsNone(reconciled["pending_at"])
        self.assertIsNone(reconciled["receipt"])
        failure, = reconciled["failed_attempts"]
        self.assertEqual(failure["attempt"], 1)
        self.assertEqual(failure["revision"], revision)
        self.assertEqual(failure["snapshot_path"], pending["snapshot_path"])
        self.assertEqual(failure["reason"], "Invalid attachment")
        self.assertEqual(failure["approved_at"], pending["approved_at"])
        self.assertEqual(failure["pending_at"], pending["pending_at"])
        self.assertEqual(Path(failure["evidence_path"]).read_bytes(), self.failure_evidence.read_bytes())
        self.rejected("No pending attempt", "failed", revision, *self.failure_args())
        self.rejected("Approve the exact revision", "ready", revision)
        self.ok("approve", revision)
        self.assertEqual(self.ok("ready", revision)["attempt"], 2)
        self.rejected("current pending attempt", "failed", revision, *self.failure_args())
        self.rejected("already used", "failed", revision, *self.failure_args(attempt=2))
        receipt_args = revision, "--success", "--confirmation", "success", "--evidence", self.evidence
        self.rejected("current pending attempt", "receipt", *receipt_args)
        self.rejected("current pending attempt", "receipt", *receipt_args, "--attempt", 1)
        self.failure_evidence.unlink()
        submitted = self.submit_receipt(revision, "--attempt", 2)
        self.assertEqual(submitted["status"], "submitted")
        self.assertEqual(submitted["receipt"]["attempt"], 2)
        self.assertEqual(submitted["failed_attempts"], [failure])
        self.assertEqual(self.ok("status"), submitted)
        self.rejected("already recorded", "failed", revision, *self.failure_args(attempt=2))

    def test_failed_history_survives_snapshot_of_corrected_packet(self):
        revision = self.snapshot()["revision"]
        self.ok("approve", revision)
        self.ok("ready", revision)
        self.attachment.write_bytes(b"Corrected synthetic attachment")
        reconciled = self.ok("failed", revision, *self.failure_args())
        self.assertEqual(reconciled["status"], "changed")
        self.rejected("input changed", "approve", revision)
        corrected = self.snapshot()
        self.assertNotEqual(corrected["revision"], revision)
        self.assertEqual(corrected["failed_attempts"], reconciled["failed_attempts"])
        self.rejected("Approve the exact revision", "ready", corrected["revision"])
        self.ok("approve", corrected["revision"])
        self.assertEqual(self.ok("ready", corrected["revision"])["attempt"], 2)
        self.failure_evidence.write_text("Test portal: attempt 2 rejected and nothing was submitted.\n")
        outcomes = self.ok("failed", corrected["revision"], *self.failure_args(attempt=2))["failed_attempts"]
        self.assertEqual([attempt["attempt"] for attempt in outcomes], [1, 2])


if __name__ == "__main__":
    unittest.main()
