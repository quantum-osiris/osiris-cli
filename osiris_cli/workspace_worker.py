"""Process one hosted Mission Control proposal on the operator's local worker."""

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

from osiris_cli.orchestration import council
from osiris_cli.providers.ollama_compat import ollama_base_url
from osiris_cli.workspace_store import StoreError, WorkspaceStore


def require_local_ollama() -> None:
    try:
        url = urlsplit(ollama_base_url())
        loopback = url.hostname == "localhost" or ipaddress.ip_address(url.hostname or "").is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise StoreError("BAD_CONFIG", "the hosted workspace worker requires loopback OLLAMA_HOST")


def process_request(store: WorkspaceStore, receipts_dir: Path, request_id: str | None) -> dict:
    require_local_ollama()
    claim = store.claim_proposal_request(request_id)
    if claim is None:
        return {"status": "NO_REQUEST_AVAILABLE"}

    request_id = claim["id"]
    lease_id = claim["lease_id"]
    receipts_path = receipts_dir / request_id / f"{lease_id}.jsonl"
    result = council.run_task(claim["objective"], receipts_path=str(receipts_path), allow_cloud=False)
    saved = store.finish_proposal_request(request_id, lease_id, result)
    return {
        "status": "SAVED",
        "request_id": request_id,
        "decision": saved["decision"],
        "receipt_hash": saved["receipt_hash"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="Claim this request UUID. The default claims the oldest eligible request.")
    parser.add_argument("--receipts-dir", type=Path, default=Path.home() / ".osiris" / "workspace-receipts")
    parser.add_argument("--local", action="store_true",
                        help="Use the loopback stand-in (OSIRIS_SUPABASE_LOCAL_URL, OSIRIS_SUPABASE_LOCAL_SECRET_KEY) "
                             "instead of the hosted project.")
    args = parser.parse_args()
    try:
        store = WorkspaceStore.local() if args.local else WorkspaceStore()
        result = process_request(store, args.receipts_dir, args.request)
    except (StoreError, council.CouncilError, council.ReceiptError) as error:
        code = error.code if isinstance(error, StoreError) else type(error).__name__
        print(json.dumps({"status": "WORKER_ERROR", "code": code}), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
