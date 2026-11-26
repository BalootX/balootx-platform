"""Require an App-recorded review for changes to critical paths."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys


POLICY_PATH = "ci/critical-paths.json"
SHA = r"[0-9a-f]{40}"
VERDICT = re.compile(rf"VERDICT: (GO|NO_GO) ({SHA})")
QUEUE_ENTRY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      headRefOid
      mergeQueueEntry { baseCommit { oid } headCommit { oid } }
    }
  }
}
"""


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, timeout=10).stdout


def pull_request_merge(read_git, head):
    parents = read_git("rev-list", "--parents", "-n", "1", "HEAD").decode("ascii").split()
    if len(parents) != 3 or parents[2] != head:
        raise ValueError(
            "Expected a two-parent checkout with the event's PR head as the second parent. "
            "Update the PR branch to start a fresh CI run."
        )
    return parents[0], parents[1]


def github(*args):
    return json.loads(subprocess.run(["gh", "api", *args], check=True, capture_output=True, timeout=30).stdout)


def policy(commit):
    config = json.loads(git("cat-file", "blob", f"{commit}:{POLICY_PATH}"))
    if type(config["schemaVersion"]) is not int or config["schemaVersion"] != 1:
        raise ValueError(f"Unsupported critical-path schema version at {commit}")
    reviewer_ids, classes = config["reviewerAccountIds"], config["classes"]
    if not isinstance(reviewer_ids, list) or not reviewer_ids or any(
        type(account_id) is not int or account_id <= 0 for account_id in reviewer_ids
    ):
        raise ValueError(f"Invalid reviewerAccountIds at {commit}")
    if not isinstance(classes, dict) or not classes:
        raise ValueError(f"Missing critical-path classes at {commit}")
    globs = set()
    for name, patterns in classes.items():
        if not name or not isinstance(patterns, list) or not patterns or any(
            not isinstance(pattern, str) or not re.fullmatch(r"[\w.*-]+(?:/[\w.*-]+)*", pattern, re.ASCII)
            or {".", ".."} & set(pattern.split("/")) for pattern in patterns
        ):
            raise ValueError(f"Invalid critical-path globs in class {name!r} at {commit}")
        globs.update(patterns)
    if ".github/workflows/**" not in globs:
        raise ValueError(f"Missing critical workflow paths at {commit}")
    patterns = [
        re.compile(
            re.escape(pattern).replace(r"\*\*/", "(?:.*/)?").replace(r"\*\*", ".*").replace(r"\*", "[^/]*"),
            re.DOTALL,
        )
        for pattern in sorted(globs)
    ]
    return set(reviewer_ids), patterns


def queue_pr_number(group, ref, candidate):
    branch = group["base_ref"].removeprefix("refs/heads/")
    pattern = rf"refs/heads/gh-readonly-queue/{re.escape(branch)}/pr-([1-9][0-9]*)-{SHA}"
    match = re.fullmatch(pattern, group["head_ref"])
    if match is None or group["head_ref"] != ref or group["head_sha"] != candidate:
        raise ValueError("Merge-group ref does not identify the event's PR, base and candidate")
    return int(match[1])


def queued_pr_head(repo, number, base, candidate):
    owner, name = repo.split("/", 1)
    response = github(
        "graphql", "-f", f"query={QUEUE_ENTRY}", "-f", f"owner={owner}", "-f", f"name={name}",
        "-F", f"number={number}",
    )
    pr = response["data"]["repository"]["pullRequest"]
    entry = pr["mergeQueueEntry"]
    if entry is None or entry["headCommit"]["oid"] != candidate or entry["baseCommit"]["oid"] != base:
        raise ValueError(f"PR #{number} no longer has the event's merge-queue entry")
    head = pr["headRefOid"]
    if not re.fullmatch(SHA, head):
        raise ValueError(f"PR #{number} has no full head SHA")
    return head


def reviewed_sha(comments, reviewer_ids, number):
    for comment in comments:
        author = comment["user"]
        if author is None or author["id"] not in reviewer_ids:
            continue
        lines = [line for line in (comment["body"] or "").splitlines() if line.lstrip().startswith("VERDICT:")]
        if not lines:
            continue
        match = VERDICT.fullmatch(lines[0])
        if len(lines) != 1 or match is None:
            raise ValueError(f"PR #{number} has a malformed VERDICT line in comment {comment['id']}")
        if match[1] != "GO":
            raise ValueError(f"PR #{number} has a NO_GO verdict on {match[2]}")
        return match[2]
    raise ValueError(f"PR #{number} needs a VERDICT: GO from a configured reviewer")


def pr_snapshot(endpoint, number):
    pr = github(endpoint)
    head, base, count = pr["head"]["sha"], pr["base"]["sha"], pr["changed_files"]
    if (pr["number"] != number or pr["state"] != "open"
            or not all(re.fullmatch(SHA, sha) for sha in (head, base))
            or type(count) is not int or not 0 <= count <= 3000):
        raise ValueError(f"PR #{number} has no complete changed-file snapshot")
    return head, base, count


def critical_paths(repo, number, head, patterns, expected_base=None):
    endpoint = f"repos/{repo}/pulls/{number}"
    snapshot = pr_snapshot(endpoint, number)
    if snapshot[0] != head or (expected_base is not None and snapshot[1] != expected_base):
        raise ValueError(f"PR #{number} no longer matches the event's head or base")
    pages = github("--paginate", "--slurp", f"{endpoint}/files?per_page=100")
    files = [item for page in pages for item in page]
    filenames = {item["filename"] for item in files}
    if snapshot[2] != len(files) or len(files) != len(filenames) or pr_snapshot(endpoint, number) != snapshot:
        raise ValueError(f"PR #{number} changed-file snapshot is incomplete or has moved")

    # Both sides of a rename are relevant when a file moves into or out of scope.
    filenames.update(item["previous_filename"] for item in files if item["status"] == "renamed")
    for path in filenames:
        if not isinstance(path, str) or "\0" in path or {"", ".", ".."} & set(path.split("/")):
            raise ValueError(f"PR #{number} has an invalid changed-file path")
    return snapshot[1], {
        path.encode("utf-8") for path in filenames if any(pattern.fullmatch(path) for pattern in patterns)
    }


def critical_entries(commit, paths):
    try:
        kind = git("cat-file", "-t", commit)
    except subprocess.CalledProcessError:
        git("fetch", "--no-tags", "--no-recurse-submodules", "--depth=1", "origin", commit)
        kind = git("cat-file", "-t", commit)
    if kind.strip() != b"commit":
        raise ValueError(f"Review comparison needs a commit at {commit}")
    entries = {}
    for line in git("ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0")[:-1]:
        identity, _, path = line.partition(b"\t")
        if path in paths:
            entries[path] = identity
    return entries


def evaluate(event_name, event, repo, ref, candidate, *, comments=None):
    if event_name == "pull_request":
        pr = event["pull_request"]
        number, base, head = pr["number"], pr["base"]["sha"], pr["head"]["sha"]
        checkout, policy_base = pull_request_merge(git, head)
        if checkout != candidate:
            raise ValueError("Expected the workflow's PR merge commit. Update the PR branch to start a fresh CI run.")
    elif event_name == "merge_group":
        group = event["merge_group"]
        base, head = group["base_sha"], None
        policy_base = base
        number = queue_pr_number(group, ref, candidate)
        if git("rev-list", "--parents", "-n", "1", "HEAD").decode("ascii").split() != [candidate, base]:
            raise ValueError("Checkout does not match the merge group commit and its base parent")
    else:
        raise ValueError(f"Unsupported review gate event {event_name}")
    if not all(re.fullmatch(SHA, sha) for sha in (candidate, base, policy_base)):
        raise ValueError("Missing full commit identity for critical-path comparison")
    if type(number) is not int or number <= 0:
        raise ValueError("Missing PR number")

    reviewer_ids, patterns = policy(policy_base)
    if event_name == "merge_group":
        head = queued_pr_head(repo, number, base, candidate)
    pr_base, critical_changes = critical_paths(repo, number, head, patterns, base if event_name == "pull_request" else None)
    if not critical_changes:
        return f"No critical paths changed (policy {policy_base[:12]})."

    head_entries = critical_entries(head, critical_changes)
    missing = critical_changes.difference(head_entries)
    if missing:
        unresolved = missing.difference(critical_entries(pr_base, missing))
        if unresolved:
            raise ValueError(
                f"PR #{number} lists {len(unresolved)} critical changed paths absent from both the PR base and head"
            )

    if comments is None:
        pages = github("--paginate", "--slurp", f"repos/{repo}/issues/{number}/comments?per_page=100")
        comments = [comment for page in pages for comment in page]
    # GitHub lists issue comments in ascending ID order across pages.
    reviewed = reviewed_sha(reversed(comments), reviewer_ids, number)
    reviewed_entries = head_entries if reviewed == head else critical_entries(reviewed, critical_changes)
    if head_entries != reviewed_entries:
        raise ValueError(f"PR #{number} has changed or missing critical paths relative to the GO on {reviewed}")
    return f"PR #{number} head {head} is covered by the GO on {reviewed}."


def check(event_name, event, repo, ref, candidate):
    print(evaluate(event_name, event, repo, ref, candidate))


if __name__ == "__main__":
    try:
        check(
            os.environ["GITHUB_EVENT_NAME"], json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes()),
            os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_REF"], os.environ["GITHUB_SHA"],
        )
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        sys.exit(f"Critical review gate failed: {exc}")
