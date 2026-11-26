"""Publish the base-owned review decision from a dedicated bare repository."""

import json
import os
import subprocess
import time


CONTEXT = "review-verdict"
QUERY = """
query($owner: String!, $name: String!, $branch: String!, $endCursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, baseRefName: $branch, first: 100, after: $endCursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number headRefOid baseRefOid
        comments(last: 1) { totalCount nodes { id updatedAt } }
        mergeQueueEntry { baseCommit { oid } headCommit { oid } }
      }
    }
  }
}
"""


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


def decision(repo, branch, pr, main_ref, ref, candidate, queue=False):
    number, head = pr["number"], pr["headRefOid"]
    try:
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
            event = {"pull_request": {"number": number, "base": {"sha": base}, "head": {"sha": head}}}

        # An open PR can be retargeted while its GitHub-generated ref is fetched.
        git("merge-base", "--is-ancestor", base, main_ref)
        git("update-ref", "--no-deref", "HEAD", candidate)
        return "success", load_review(base)(event_name, event, repo, ref, candidate)
    except json.JSONDecodeError as exc:
        return "pending", f"Verdict not re-checked: {type(exc).__name__}"
    except ValueError as exc:
        return "failure", str(exc)
    except Exception as exc:
        return "pending", f"Verdict not re-checked: {type(exc).__name__}"


def publish(repo, public_url, bot_id, number, target, state, reason):
    endpoint = f"repos/{repo}"
    pages = github("--paginate", "--slurp", f"{endpoint}/commits/{target}/statuses?per_page=100")
    previous = next((
        row["state"] for page in pages for row in page
        if row["context"].casefold() == CONTEXT and (row["creator"] or {}).get("id") == bot_id
    ), None)
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


def targets(repo, branch, main_ref):
    owner, name = repo.split("/", 1)
    pages = github("--paginate", "--slurp", "graphql", "-f", f"query={QUERY}",
                   "-f", f"owner={owner}", "-f", f"name={name}", "-f", f"branch={branch}")
    prs = [pr for page in pages for pr in page["data"]["repository"]["pullRequests"]["nodes"]]
    branch_ref = f"refs/heads/{branch}"
    queue_prefix = f"refs/heads/gh-readonly-queue/{branch}/"
    refs = {ref: oid for oid, ref in (
        line.split("\t") for line in git(
            "ls-remote", "--refs", "origin", branch_ref, "refs/pull/*/merge", f"{queue_prefix}*",
        ).decode("ascii").splitlines()
    )}
    if git("rev-parse", main_ref).decode("ascii").strip() != refs[branch_ref]:
        git("fetch", "--no-tags", "--no-recurse-submodules", "origin", f"{branch_ref}:{main_ref}")
    found = {}
    for pr in prs:
        number, head = pr["number"], pr["headRefOid"]
        ref = f"refs/pull/{number}/merge"
        found.setdefault(head, []).append((pr, ref, refs.get(ref), False))
        entry = pr["mergeQueueEntry"]
        if entry and entry["headCommit"] and entry["baseCommit"]:
            queued = entry["headCommit"]["oid"]
            matches = [ref for ref, oid in refs.items() if oid == queued and ref.startswith(f"{queue_prefix}pr-{number}-")]
            ref, candidate = (matches[0], queued) if len(matches) == 1 else (None, None)
            found.setdefault(queued, []).append((pr, ref, candidate, True))
    return found


def poll(repo, branch, public_url, bot_id, seconds):
    main_ref = f"refs/heads/{branch}"
    seen, current = {}, {}
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        started = time.monotonic()
        try:
            current = targets(repo, branch, main_ref)
            outage = None
        except Exception as exc:
            outage = f"Verdict not re-checked: {type(exc).__name__}"
            print(outage, flush=True)
        seen = {sha: key for sha, key in seen.items() if sha in current}
        for sha, entries in current.items():
            number = entries[0][0]["number"]
            if outage:
                state, reason = "pending", outage
            elif entries == seen.get(sha):
                continue
            elif len(entries) != 1:
                state, reason = "failure", "Commit is associated with multiple open review targets"
            else:
                pr, ref, candidate, queue = entries[0]
                state, reason = decision(repo, branch, pr, main_ref, ref, candidate, queue=queue)
            try:
                publish(repo, public_url, bot_id, number, sha, state, reason)
                if state != "pending":
                    seen[sha] = entries
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
