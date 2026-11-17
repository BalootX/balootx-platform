from __future__ import annotations

import argparse
from pathlib import Path

from baloot_platform.config import build_bundle
from baloot_platform.guide import build_engine_guide


ACTIVE_GUIDE_BUNDLE = Path("configs/bundle_ortools_research_native_v3_1.yaml")


def render_markdown(guide: dict) -> str:
    lines = [
        f"# {guide['pageTitle']}",
        "",
        guide["subtitle"],
        "",
        f"> **{guide['proofBoundary']['title']}:** {guide['proofBoundary']['body']}",
        "",
        f"**{guide['profile']['label']}:** `{guide['profile']['id']}` · "
        f"{guide['interface']['versionPrefix']} `{guide['profile']['version']}` · "
        f"{guide['profile']['status']}",
        "",
    ]
    for section in guide["sections"]:
        lines.extend([f"## {section['title']}", "", section["summary"], ""])
        for fact in section["facts"]:
            lines.append(f"- **{fact['label']}:** {fact['value']}")
        for table in section["tables"]:
            lines.extend([
                "",
                f"### {table['title']}",
                "",
                f"| {guide['interface']['markdownCardColumn']} | "
                f"{guide['interface']['markdownPointsColumn']} |",
                "|---|---:|",
            ])
            lines.extend(f"| {card['rank']} | {card['points']} |" for card in table["cards"])
        lines.extend(["", f"### {guide['interface']['examplesLabel']}", ""])
        lines.extend(f"- {example}" for example in section["examples"])
        lines.append("")
    lines.extend(
        [
            f"## {guide['interface']['markdownAuditHeading']}",
            "",
            f"`{guide['guideHash']}`",
            "",
            guide["interface"]["markdownGeneratedNotice"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=ACTIVE_GUIDE_BUNDLE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    guide = build_engine_guide(
        build_bundle(root, bundle_path=args.bundle),
        root=root,
    )
    target = args.output or root / "docs" / "ENGINE_GUIDE_AR.md"
    target.write_text(render_markdown(guide), encoding="utf-8", newline="\n")
    print(target)


if __name__ == "__main__":
    main()
