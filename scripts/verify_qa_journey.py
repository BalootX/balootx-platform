"""Require complete Go and Foundry evidence for a hosted QA journey."""

from collections import defaultdict
from glob import glob
import json
from pathlib import Path
import sys
from xml.etree import ElementTree


GO_PACKAGE = "github.com/example/baloot-platform-sdk-go"
GO_TESTS = (
    "TestResearchBundleSelectorsPreserveLegacyReads",
    "TestResearchRenderRequestsForwardObserverSeat",
    "TestResearchRecipeMutationsGenerateOnlyAbsentIdempotencyKeys",
    "TestResearchRecipeListCursorAndTypedPagePreserveLegacyCallers",
    "TestResearchResultOperatorAuditDecodesTypedResponse",
    "TestResearchResultFiltersCoverV4SemanticFacets",
)
FOUNDRY_PROJECTS = {"desktop-chrome", "mobile-touch-chrome"}
FOUNDRY_SPEC = "foundry_ortools_operator.spec.ts"
FOUNDRY_TEST = (
    "ACC-19 exact request and disclosed alternative › "
    "the disclosed target-wins-trick-6 alternative diagnoses its first result"
)
FOUNDRY_SCREENSHOTS = {
    f"alternate-first-result-{view}.png" for view in ("estimate", "terminal-run", "terminal-result")
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def verify_go(receipt: Path) -> None:
    phases = defaultdict(list)
    with receipt.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            action = event["Action"]
            test = event.get("Test")
            if event["Package"] != GO_PACKAGE or action in {"fail", "skip"}:
                raise ValueError("Go SDK package or test failed to complete successfully")
            if action == "output" and test is None and event["Output"].rstrip().endswith("\t(cached)"):
                raise ValueError("Go SDK receipt contains cached results")
            if action in {"start", "run", "pass"}:
                phases[test].append(action)
    for name in GO_TESTS:
        if phases.get(name) != ["run", "pass"]:
            raise ValueError(f"Go SDK receipt requires one run and pass for {name}")
    if phases.get(None) != ["start", "pass"] or action != "pass" or test is not None:
        raise ValueError("Go SDK receipt requires one final package pass")


def _passed_count(group: ElementTree.Element, count: int) -> bool:
    return group.get("tests") == str(count) and all(group.get(field) == "0" for field in ("skipped", "errors", "failures"))


def verify_foundry(receipt: Path, screenshot_glob: str) -> None:
    report = ElementTree.parse(receipt).getroot()
    suites = report.findall("testsuite")
    if (report.tag != "testsuites" or len(suites) != len(FOUNDRY_PROJECTS)
            or {suite.get("hostname") for suite in suites} != FOUNDRY_PROJECTS):
        raise ValueError("Foundry receipt requires one suite for each journey project")
    if not _passed_count(report, len(FOUNDRY_PROJECTS)) or not all(_passed_count(suite, 1) for suite in suites):
        raise ValueError("Foundry receipt requires complete passing browser results")
    if (len(report.findall(".//testcase")) != len(FOUNDRY_PROJECTS)
            or any(report.find(f".//{tag}") is not None for tag in ("skipped", "error", "failure"))):
        raise ValueError("Foundry receipt contains an incomplete browser result")
    for suite in suites:
        cases = suite.findall("testcase")
        if (len(cases) != 1 or cases[0].get("name") != FOUNDRY_TEST
                or cases[0].get("classname") != FOUNDRY_SPEC):
            raise ValueError("Foundry receipt requires the named journey in each browser project")

    screenshots = [Path(path) for path in glob(screenshot_glob)]
    if len(screenshots) != len(FOUNDRY_PROJECTS) * len(FOUNDRY_SCREENSHOTS):
        raise ValueError("Foundry requires three screenshots per browser project")
    for project in sorted(FOUNDRY_PROJECTS):
        views = [path for path in screenshots if path.parent.name.endswith(f"-{project}")]
        if ({path.name for path in views} != FOUNDRY_SCREENSHOTS
                or len({path.parent for path in views}) != 1):
            raise ValueError(f"Foundry screenshot set differs for {project}")
    for path in screenshots:
        with path.open("rb") as image:
            if image.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE or not image.read(1):
                raise ValueError(f"Foundry screenshot is not a PNG: {path.name}")


if __name__ == "__main__":
    try:
        match sys.argv[1:]:
            case ["go", receipt]:
                verify_go(Path(receipt))
            case ["foundry", receipt, screenshot_glob]:
                verify_foundry(Path(receipt), screenshot_glob)
            case _:
                raise ValueError("expected go <jsonl> or foundry <junit> <screenshot glob>")
    except (OSError, ValueError, KeyError, TypeError, ElementTree.ParseError) as error:
        raise SystemExit(f"QA journey evidence rejected: {error}") from None
