"""Preserve public interface metadata and license evidence; never execute upstream code."""
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import subprocess

REPOSITORIES = (
    "MattBryantt/fpl", "Shaqthegr8/fplanner", "jotohow/lionel",
    "daniel-mehta/fpl-forecast", "umer33511/fpl_manager_agent",
)


def api(endpoint):
    """Read a public GitHub API resource with the installed authenticated CLI."""
    response = subprocess.run(["gh", "api", endpoint], check=True, text=True, capture_output=True)
    return json.loads(response.stdout)


def main():
    root = Path(__file__).resolve().parent
    output = root / "inventory.json"
    if output.exists():
        raise ValueError("Preserve previous inventory; no overwrite")
    inventory = {"status": "complete", "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                 "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "repositories": {}, "limitations": [
                     "GitHub license classification is a clue; actual notices must be reviewed.",
                     "Code licensing does not independently establish permission for external club/player assets.",
                     "Metadata and source are inspected only; no upstream code is executed or adopted.",
                     "This interface audit is separate from forecasting data admission."]}
    for name in REPOSITORIES:
        metadata = api(f"repos/{name}")
        commit = api(f"repos/{name}/commits/{metadata['default_branch']}")
        revision = commit["sha"]
        tree = api(f"repos/{name}/git/trees/{revision}?recursive=1")
        if tree.get("truncated"):
            raise ValueError(f"Incomplete upstream tree: {name}")
        paths = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
        selected = [p for p in paths if "/" not in p and p.lower() in {
            "license", "license.md", "license.txt", "copying", "copying.md", "copying.txt", "readme.md"}]
        raw_root = root / "raw" / name.replace("/", "__")
        raw_root.mkdir(parents=True, exist_ok=True)
        sources = {}
        for path in selected:
            blob = api(f"repos/{name}/git/blobs/{paths[path]['sha']}")
            if blob["encoding"] != "base64" or blob["size"] > 500_000:
                raise ValueError(f"Unexpected text source encoding/size: {name}/{path}")
            body = base64.b64decode(blob["content"])
            git_hash = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
            if len(body) != blob["size"] or git_hash != paths[path]["sha"]:
                raise ValueError(f"Upstream blob identity mismatch: {name}/{path}")
            destination = raw_root / path
            if destination.exists():
                raise ValueError(f"Preserve raw evidence: {destination}")
            destination.write_bytes(body)
            sources[path] = {"sha256": hashlib.sha256(body).hexdigest(), "git_blob_sha": git_hash,
                             "url": f"https://github.com/{name}/blob/{revision}/{path}",
                             "local_path": str(destination.relative_to(root))}
        inventory["repositories"][name] = {
            "revision": revision, "commit_date": commit["commit"]["committer"]["date"],
            "default_branch": metadata["default_branch"], "archived": metadata["archived"],
            "github_license_id": (metadata.get("license") or {}).get("spdx_id"),
            "pushed_at": metadata["pushed_at"], "source_url": metadata["html_url"],
            "tree_sha": tree["sha"], "tree_files": len(paths), "sources": sources,
            "interface_paths": [p for p in paths if p.endswith((".py", ".tsx", ".ts", ".html", ".mjs"))
                                and any(word in p.lower() for word in ("streamlit", "frontend/src", "web/", "app.py", "server.py"))][:100]}
    with output.open("x") as handle:
        json.dump(inventory, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({k: {"revision": v["revision"], "license": v["github_license_id"],
                         "license_files": [p for p in v["sources"] if not p.lower().startswith("readme")]}
                      for k, v in inventory["repositories"].items()}, indent=2))


if __name__ == "__main__":
    main()
