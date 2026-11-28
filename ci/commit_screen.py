"""Screen Git commit identities against the event's base policy."""

from collections import Counter
from itertools import batched
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


POLICY_PATH = "ci/commit-identities.json"
SCHEMA_PATH = "schemas/contracts/commit-identities-v1.schema.json"
SHA = re.compile(r"[0-9a-f]{40}")
ZERO = "0" * 40
EMAIL = re.compile(r"[\w.!#$%&'*+/=?^_{|}~\[\]-]+@[\w-]+(?:\.[\w-]+)+")
IDENTITY_TRAILERS = {"co-authored-by": "Co-authored-by", "signed-off-by": "Signed-off-by"}
IDENTITY_KEYS = {"author", "co-author", "committer", "reviewer", "signer", "cc", "bcc", "to"}
COMMIT_FORMAT = (
    "%H%x00%an%x00%ae%x00%cn%x00%ce%x00%B%x00"
    "%(trailers:only,unfold,separator=%x0a,key_value_separator=%x3a%x20)"
)


class ScreenError(ValueError):
    """A diagnostic safe to print without commit contents."""


def git(*args):
    return subprocess.run(
        ["git", "--no-replace-objects", *args], check=True, capture_output=True, timeout=30
    ).stdout


def commit_sha(value):
    if not isinstance(value, str) or value == ZERO or not SHA.fullmatch(value):
        raise ScreenError("The event needs a full commit SHA.")
    return value


def new_branch_revisions(pushed, base, candidate):
    if len(pushed) >= 2048:
        raise ScreenError("The new branch push needs a complete commit list below the GitHub payload limit.")
    if len(pushed) != len(set(pushed)) or (pushed and candidate not in pushed):
        raise ScreenError("The new branch commit list needs unique IDs and must include the pushed head.")
    if git("rev-parse", "--is-shallow-repository").strip() != b"false":
        raise ScreenError("The new branch commit list requires complete Git history.")
    required = set(git("rev-list", f"{base}..{candidate}").decode("ascii").split())
    if required.difference(pushed):
        raise ScreenError("The new branch commit list must cover all commits outside the trusted base history.")
    if not pushed:
        return []
    try:
        listed = git("rev-list", "--no-walk=unsorted", *pushed).decode("ascii").split()
    except subprocess.CalledProcessError as exc:
        raise ScreenError("The new branch commit IDs must identify commit objects.") from exc
    if set(listed) != set(pushed):
        raise ScreenError("The new branch commit IDs must identify commit objects.")
    if git("rev-list", "--max-count=1", *pushed, f"^{candidate}"):
        raise ScreenError("The new branch commit list contains a commit outside the pushed head history.")
    return ["--no-walk=unsorted", *pushed]


def push_scope(event, candidate):
    ref = event.get("ref")
    repository = event.get("repository")
    default = repository.get("default_branch") if isinstance(repository, dict) else None
    if not isinstance(ref, str) or not ref.startswith("refs/heads/") or not isinstance(default, str) or not default:
        raise ScreenError("The push event needs a branch ref and a default branch.")
    default_ref = f"refs/heads/{default}"
    try:
        git("check-ref-format", ref)
        git("check-ref-format", default_ref)
    except subprocess.CalledProcessError as exc:
        raise ScreenError("The push event needs valid branch refs.") from exc

    before, after = event.get("before"), event.get("after")
    created, deleted, forced = (event.get(field) for field in ("created", "deleted", "forced"))
    if before != ZERO:
        commit_sha(before)
    if after != ZERO:
        commit_sha(after)
    if (any(type(flag) is not bool for flag in (created, deleted, forced))
            or created != (before == ZERO) or deleted != (after == ZERO) or (created and deleted)
            or ((created or deleted) and ref == default_ref)):
        raise ScreenError("The push event needs consistent branch creation and deletion metadata.")

    commits = event.get("commits")
    if not isinstance(commits, list) or not all(isinstance(row, dict) and "id" in row for row in commits):
        raise ScreenError("The push event needs a commit list containing commit IDs.")
    pushed = [commit_sha(row["id"]) for row in commits]
    head = event.get("head_commit")
    if "head_commit" not in event or (head is not None and (not isinstance(head, dict) or head.get("id") != after)):
        raise ScreenError("The push event needs consistent head commit metadata.")
    if deleted:
        if pushed or head is not None or forced:
            raise ScreenError("A branch deletion must have no pushed commits, head commit, or force flag.")
        return None, []
    if pushed and head is None:
        raise ScreenError("The push event needs consistent head commit metadata.")
    if after != candidate:
        raise ScreenError("The checkout must match the pushed head.")

    if ref == default_ref:
        # A push to the default branch uses its pre-push policy.
        base = commit_sha(before)
    else:
        base = commit_sha(git("rev-parse", "--verify", f"refs/remotes/origin/{default}^{{commit}}").decode().strip())

    if created:
        return base, new_branch_revisions(pushed, base, candidate)

    if git("rev-parse", "--is-shallow-repository").strip() != b"false":
        raise ScreenError("A branch update requires complete Git history.")
    try:
        old_kind = git("cat-file", "-t", before).strip()
    except subprocess.CalledProcessError:
        # Force pushes can leave the previous tip outside fetched refs.
        git("fetch", "--no-tags", "--no-recurse-submodules", "origin", before)
        old_kind = git("cat-file", "-t", before).strip()
    if old_kind != b"commit":
        raise ScreenError("The branch update's previous tip must be a commit object.")
    if not forced:
        try:
            git("merge-base", "--is-ancestor", before, candidate)
        except subprocess.CalledProcessError as exc:
            raise ScreenError("A non-forced branch update must be a fast-forward.") from exc

    update_range = f"{before}..{candidate}"
    actual = set(git("rev-list", update_range).decode("ascii").split())
    supplied = set(pushed)
    if len(supplied) != len(pushed) or not supplied <= actual:
        raise ScreenError("The branch update commit list needs unique IDs within the Git update range.")
    return base, [update_range]


def event_scope(event_name, event, candidate):
    """Return the policy commit and the git-log selection for this event."""
    candidate = commit_sha(candidate)
    parents = git("rev-list", "--parents", "-n", "1", "HEAD").decode("ascii").split()
    if not parents or parents[0] != candidate:
        raise ScreenError("The checkout must match the event commit.")

    if event_name == "pull_request":
        pr = event["pull_request"]
        base, head = commit_sha(pr["base"]["sha"]), commit_sha(pr["head"]["sha"])
        if parents != [candidate, base, head]:
            raise ScreenError("The checkout must match both pull request merge parents.")
        return base, [f"{base}..{head}"]

    if event_name == "merge_group":
        group = event["merge_group"]
        base = commit_sha(group["base_sha"])
        if candidate != commit_sha(group["head_sha"]) or base == candidate:
            raise ScreenError("The checkout must match the merge group head and follow its base.")
        git("merge-base", "--is-ancestor", base, candidate)
        return base, [f"{base}..{candidate}"]

    if event_name == "push":
        return push_scope(event, candidate)
    raise ScreenError("The checkout must match a supported screening event.")


def load_policy(base):
    try:
        policy = json.loads(git("show", f"{base}:{POLICY_PATH}"))
        schema = json.loads(git("show", f"{base}:{SCHEMA_PATH}"))
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(policy)
    except (ValueError, SchemaError, ValidationError, subprocess.SubprocessError) as exc:
        raise ScreenError(f"Commit {base} needs a valid base commit identity policy and schema.") from exc
    rows = policy["identities"]
    identities = {(row["name"], row["email"]): row["use"] for row in rows}
    if len(identities) != len(rows):
        raise ScreenError(f"Commit {base} needs unique commit identity pairs in its policy.")
    openers = policy["pullRequestOpenerIds"]
    if any(type(account_id) is not int for account_id in openers):
        raise ScreenError(f"Commit {base} needs integer pull request opener IDs in its policy.")
    return identities, set(openers)


def screen(event_name, event, candidate):
    base, revisions = event_scope(event_name, event, candidate)
    if base is None:
        print("Screened 0 commits. The branch was deleted.")
        return
    identities, approved_openers = load_policy(base)
    if event_name == "pull_request":
        opener = event["pull_request"].get("user")
        if (not isinstance(opener, dict) or opener.get("type") != "Bot"
                or type(opener.get("id")) is not int or opener["id"] not in approved_openers):
            raise ScreenError("Open this pull request through an approved GitHub App.")
    allowed_addresses = {email for (_, email), use in identities.items() if use == "all"}
    trailer_identities = {f"{name} <{email}>": (email, use) for (name, email), use in identities.items()}
    fields = []
    if revisions:
        raw = git("log", "-z", "--encoding=UTF-8", f"--format=tformat:{COMMIT_FORMAT}", *revisions, "--")
        if raw:
            if not raw.endswith(b"\0"):
                raise ScreenError("Git did not return complete commit records.")
            fields = raw.decode("utf-8").split("\0")[:-1]
    failures = []
    for sha, author, author_email, committer, committer_email, message, trailers in batched(fields, 7):
        commit_sha(sha)
        for field, pair in (("author", (author, author_email)), ("committer", (committer, committer_email))):
            if identities.get(pair) != "all":
                failures.append(f"Commit {sha} {field}: use a listed identity permitted for this field.")
        message_addresses = Counter(EMAIL.findall(message))
        for index, trailer in enumerate(trailers.split("\n") if trailers else (), 1):
            key, separator, value = trailer.partition(": ")
            if not key or not separator:
                raise ScreenError(f"Commit {sha} has an invalid Git trailer record.")
            key = key.casefold()
            known = IDENTITY_TRAILERS.get(key)
            identifies_person = key.endswith("-by") or key in IDENTITY_KEYS
            carries_address = EMAIL.search(value) or all(char in value for char in "<@>")
            if not (identifies_person or carries_address):
                continue
            email, use = trailer_identities.get(value, (None, None))
            if use == "all":
                continue
            if use is not None and use == known and message_addresses[email] > 0:
                # Each matched trailer accounts for one occurrence in the raw message.
                message_addresses[email] -= 1
            else:
                field = known or f"trailer #{index}"
                failures.append(f"Commit {sha} {field}: use a listed identity permitted for this trailer.")
        if any(count > 0 and email not in allowed_addresses for email, count in message_addresses.items()):
            failures.append(f"Commit {sha} message: use listed no-reply addresses for attribution.")
    if failures:
        raise ScreenError("\n".join(failures))
    print(f"Screened {len(fields) // 7} commits using the policy at {base}.")


if __name__ == "__main__":
    try:
        screen(os.environ["GITHUB_EVENT_NAME"], json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes()),
               os.environ["GITHUB_SHA"])
    except ScreenError as exc:
        sys.exit(str(exc))
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError):
        sys.exit("Commit screening could not read or verify the event's Git history.")
