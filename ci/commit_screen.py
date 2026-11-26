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
COMMIT_FORMAT = "%H%x00%an%x00%ae%x00%cn%x00%ce%x00%B%x00%(trailers:only,unfold)"


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


def event_scope(event_name, event, candidate):
    """Return the policy commit and the git-log selection for this event."""
    if event_name == "push" and event["deleted"] is True and event["after"] == ZERO:
        return None, []
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

    if event_name != "push" or event["after"] != candidate or not event["ref"].startswith("refs/heads/"):
        raise ScreenError("The checkout must match a supported screening event.")

    before = event["before"]
    default = event["repository"]["default_branch"]
    if event["ref"] == f"refs/heads/{default}":
        # A push to the default branch uses its pre-push policy.
        base = commit_sha(before)
    else:
        base = commit_sha(git("rev-parse", "--verify", f"refs/remotes/origin/{default}^{{commit}}").decode().strip())

    if before == ZERO and event["created"] is True:
        commits = event["commits"]
        if len(commits) >= 2048:
            raise ScreenError("The new branch push needs a complete commit list below the GitHub payload limit.")
        pushed = [commit_sha(row["id"]) for row in commits]
        return base, ["--no-walk=unsorted", *pushed] if pushed else []

    before = commit_sha(before)
    try:
        git("cat-file", "-e", f"{before}^{{commit}}")
    except subprocess.CalledProcessError:
        # Force pushes can leave the previous tip outside fetched refs.
        git("fetch", "--no-tags", "--no-recurse-submodules", "origin", before)
    return base, [f"{before}..{candidate}"]


def load_identities(base):
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
    return identities


def screen(event_name, event, candidate):
    base, revisions = event_scope(event_name, event, candidate)
    if base is None:
        print("Screened 0 commits. The branch was deleted.")
        return
    identities = load_identities(base)
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
        for index, trailer in enumerate(trailers.splitlines(), 1):
            key, _, value = trailer.partition(":")
            known = IDENTITY_TRAILERS.get(key.strip().casefold())
            carries_address = EMAIL.search(value) or all(char in value for char in "<@>")
            if not (known or carries_address):
                continue
            email, use = trailer_identities.get(value.strip(), (None, None))
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
