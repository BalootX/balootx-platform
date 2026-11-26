"""Publish the base-owned review decision from a dedicated bare repository."""

import json
import os
import subprocess
import time


CONTEXT = "review-verdict"
TARGET_FIELDS = """
fragment ReviewTarget on PullRequest {
  number headRefOid baseRefOid
  mergeQueueEntry { baseCommit { oid } headCommit { oid } }
}
"""
QUERY = """
query($owner: String!, $name: String!, $branch: String!, $endCursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, baseRefName: $branch, first: 100, after: $endCursor) {
      pageInfo { hasNextPage endCursor }
      nodes { ...ReviewTarget }
    }
  }
}
""" + TARGET_FIELDS
COMMENTS = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      ...ReviewTarget
      state baseRefName
      comments(first: 100, after: $cursor) {
        totalCount
        pageInfo { hasNextPage endCursor }
        nodes { fullDatabaseId body author { ... on Node { id } } }
      }
    }
  }
}
""" + TARGET_FIELDS


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, timeout=10).stdout


def github(*args):
    output = subprocess.run(["gh", "api", *args], check=True, capture_output=True, timeout=30).stdout
    return json.loads(output)


def load_review(base):
    path = "ci/critical_review.py"
    scope = {"__name__": "base_critical_review"}
    exec(compile(git("show", f"{base}:{path}"), path, "exec"), scope)
    return scope["evaluate"]


def decision(repo, branch, pr, main_ref, ref, candidate, queue=False, *, comments=None):
    number = pr["number"]
    try:
        head = pr["headRefOid"]
        if not candidate:
            return "pending", f"Waiting for GitHub's {'queue commit' if queue else 'test merge'} for PR #{number}"
        git("fetch", "--no-tags", "--no-recurse-submodules", "origin", ref)
        parents = git("rev-list", "--parents", "-n", "1", "FETCH_HEAD").decode("ascii").split()
        if not parents or parents[0] != candidate:
            return "pending", f"Waiting for GitHub's current merge commit for PR #{number}"
        if queue:
            base = pr["mergeQueueEntry"]["baseCommit"]["oid"]
            event_name = "merge_group"
            event = {"merge_group": {
                "base_ref": f"refs/heads/{branch}", "base_sha": base, "head_ref": ref, "head_sha": candidate,
            }}
        else:
            if len(parents) != 3 or parents[2] != head:
                return "pending", f"Waiting for GitHub's test merge of {head[:12]}"
            base = parents[1]
            event_name = "pull_request"
            event = {"pull_request": {
                "number": number, "base": {"sha": pr["baseRefOid"]}, "head": {"sha": head},
            }}

        # An open PR can be retargeted while its GitHub-generated ref is fetched.
        git("merge-base", "--is-ancestor", base, main_ref)
        git("update-ref", "--no-deref", "HEAD", candidate)
        expected_pr = (head, pr["baseRefOid"]) if queue else None
        return "success", load_review(base)(
            event_name, event, repo, ref, candidate, comments=comments, expected_pr=expected_pr,
        )
    except json.JSONDecodeError as exc:
        return "pending", f"Verdict not re-checked: {type(exc).__name__}"
    except ValueError as exc:
        return "failure", str(exc)
    except Exception as exc:
        return "pending", f"Verdict not re-checked: {type(exc).__name__}"


def publish(repo, public_url, bot_id, number, target, state, reason):
    endpoint = f"repos/{repo}"
    try:
        pages = github("--paginate", "--slurp", f"{endpoint}/commits/{target}/statuses?per_page=100")
        previous = next((
            row["state"] for page in pages for row in page
            if row["context"].casefold() == CONTEXT and (row["creator"] or {}).get("id") == bot_id
        ), None)
    except Exception as exc:
        if state == "success":
            raise
        print(f"PR #{number} status history unavailable: {type(exc).__name__}", flush=True)
        previous = None
    if previous == state or (previous == "failure" and state == "pending"):
        return
    try:
        github(
            "--method", "POST", f"{endpoint}/statuses/{target}",
            "-f", f"context={CONTEXT}", "-f", f"state={state}", "-f", f"description={reason[:140]}",
            "-f", f"target_url={public_url}/{repo}/pull/{number}",
        )
    finally:
        time.sleep(1)


def comment_snapshot(repo, branch, pr):
    number = pr["number"]
    owner, name = repo.split("/", 1)
    problem = f"PR #{number} comment snapshot is incomplete or has moved"
    args = ("graphql", "-f", f"query={COMMENTS}", "-f", f"owner={owner}", "-f", f"name={name}",
            "-F", f"number={number}")

    def read_page(cursor=None):
        response = github(*args) if cursor is None else github(*args, "-f", f"cursor={cursor}")
        if response.get("errors"):
            raise RuntimeError(problem)
        current = response["data"]["repository"]["pullRequest"]
        if (current["state"] != "OPEN" or current["baseRefName"] != branch
                or any(current[field] != pr[field] for field in ("number", "headRefOid", "baseRefOid", "mergeQueueEntry"))):
            raise RuntimeError(problem)
        return current["comments"]

    page = read_page()
    total = page["totalCount"]
    if type(total) is not int or total < 0:
        raise RuntimeError(problem)
    expected = []
    cursors = set()
    while True:
        nodes = page["nodes"]
        if page["totalCount"] != total or len(nodes) != min(100, total - len(expected)):
            raise RuntimeError(problem)
        expected.extend(
            (row["fullDatabaseId"], row["body"], None if row["author"] is None else row["author"]["id"])
            for row in nodes
        )
        info = page["pageInfo"]
        if type(info["hasNextPage"]) is not bool or info["hasNextPage"] != (len(expected) < total):
            raise RuntimeError(problem)
        if not info["hasNextPage"]:
            break
        cursor = info["endCursor"]
        if not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise RuntimeError(problem)
        cursors.add(cursor)
        page = read_page(cursor)

    # A REST offset page can shift when a comment disappears. Compare every row with the cursor view.
    pages = github("--paginate", "--slurp", f"repos/{repo}/issues/{number}/comments?per_page=100")
    if not isinstance(pages, list) or not pages or any(not isinstance(page, list) for page in pages):
        raise RuntimeError(problem)
    comments = []
    for page in pages:
        for row in page:
            author = row["user"]
            if type(row["id"]) is not int or row["id"] <= 0 or not isinstance(row["body"], str):
                raise RuntimeError(problem)
            if author is not None and (
                type(author["id"]) is not int or author["id"] <= 0
                or not isinstance(author["node_id"], str) or not author["node_id"]
            ):
                raise RuntimeError(problem)
            comments.append({
                "id": row["id"], "body": row["body"],
                "user": None if author is None else {"id": author["id"], "node_id": author["node_id"]},
            })
    ids = [row["id"] for row in comments]
    actual = [(str(row["id"]), row["body"], (row["user"] or {}).get("node_id")) for row in comments]
    if ids != sorted(set(ids)) or actual != expected:
        raise RuntimeError(problem)
    return comments


def targets(repo, branch):
    owner, name = repo.split("/", 1)
    pages = github("--paginate", "--slurp", "graphql", "-f", f"query={QUERY}",
                   "-f", f"owner={owner}", "-f", f"name={name}", "-f", f"branch={branch}")
    if not pages or any(page.get("errors") for page in pages):
        raise RuntimeError("Open PR snapshot is incomplete")
    prs = [pr for page in pages for pr in page["data"]["repository"]["pullRequests"]["nodes"]]
    found = {}
    for pr in prs:
        if head := pr.get("headRefOid"):
            found.setdefault(head, []).append((pr, False))
        entry = pr.get("mergeQueueEntry") or {}
        if queued := (entry.get("headCommit") or {}).get("oid"):
            found.setdefault(queued, []).append((pr, True))
    return found


def merge_refs(branch, main_ref):
    branch_ref = f"refs/heads/{branch}"
    queue_prefix = f"refs/heads/gh-readonly-queue/{branch}/"
    refs = {ref: oid for oid, ref in (
        line.split("\t") for line in git(
            "ls-remote", "--refs", "origin", branch_ref, "refs/pull/*/merge", f"{queue_prefix}*",
        ).decode("ascii").splitlines()
    )}
    if git("rev-parse", main_ref).decode("ascii").strip() != refs[branch_ref]:
        git("fetch", "--no-tags", "--no-recurse-submodules", "origin", f"{branch_ref}:{main_ref}")
    return refs


def poll(repo, branch, public_url, bot_id, seconds):
    main_ref = f"refs/heads/{branch}"
    seen, current = {}, {}
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        started = time.monotonic()
        try:
            # Keep the target SHAs if fetching refs or verifying the verdict fails.
            current = targets(repo, branch)
            refs = merge_refs(branch, main_ref)
            outage = None
        except Exception as exc:
            outage = f"Verdict not re-checked: {type(exc).__name__}"
            print(outage, flush=True)
        seen = {sha: key for sha, key in seen.items() if sha in current}
        snapshots = {}
        for sha, entries in current.items():
            number = entries[0][0]["number"]
            key = None
            if outage:
                state, reason = "pending", outage
            elif len(entries) != 1:
                key = (entries, None)
                if key == seen.get(sha):
                    continue
                state, reason = "failure", "Commit is associated with multiple open review targets"
            else:
                pr, queue = entries[0]
                if number not in snapshots:
                    try:
                        snapshots[number] = (comment_snapshot(repo, branch, pr), None)
                    except Exception as exc:
                        snapshots[number] = (None, f"Comment snapshot unavailable: {type(exc).__name__}")
                comments, snapshot_error = snapshots[number]
                if snapshot_error:
                    state, reason = "pending", snapshot_error
                else:
                    if queue:
                        prefix = f"refs/heads/gh-readonly-queue/{branch}/pr-{number}-"
                        matches = [ref for ref, oid in refs.items() if oid == sha and ref.startswith(prefix)]
                        ref = matches[0] if len(matches) == 1 else None
                    else:
                        ref = f"refs/pull/{number}/merge"
                    candidate = refs.get(ref)
                    key = (entries, ref, candidate, comments)
                    if key == seen.get(sha):
                        continue
                    state, reason = decision(repo, branch, pr, main_ref, ref, candidate, queue=queue, comments=comments)
            try:
                publish(repo, public_url, bot_id, number, sha, state, reason)
                if key is not None and state != "pending":
                    seen[sha] = key
                    continue
            except Exception as exc:
                print(f"PR #{number} status unavailable: {type(exc).__name__}", flush=True)
            seen.pop(sha, None)
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(remaining, max(0, 30 - (time.monotonic() - started))))


if __name__ == "__main__":
    poll(os.environ["REVIEW_VERDICT_REPOSITORY"], os.environ["REVIEW_VERDICT_BASE_BRANCH"],
         os.environ["GITHUB_SERVER_URL"], int(os.environ["REVIEW_VERDICT_BOT_ID"]), 9 * 60)
