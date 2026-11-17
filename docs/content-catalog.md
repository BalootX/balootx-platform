# Governed synthetic clip inputs

`scripts.render_research_notice.select_deal()` resolves a checked-in fixture by
`--deal-id`. Use the explicit ID when an input must remain pinned: `--seed`
selects from the current catalog, so extending the catalog changes seed choices.
The existing default is unchanged.

The four `sun-single-winner-20261118-01-v1` through
`sun-single-winner-20261118-04-v1` inputs contain new complete deals and legal
prefixes, exact root values, a principal variation to the final score, and
expected project and card points. Their complete deals differ from the earlier
catalog even after suit and seat renaming. Their opener hands and tables also
differ from the earlier catalog.

| Suffix | Winning action | Tactical sequence in the complete state |
| --- | --- | --- |
| 01 | DQ | Bank the queen's points in partner's winning spade trick; retain the club ten for the last trick. |
| 02 | HA | Capture the heart king, then cash the spade ten before returning the remaining heart. |
| 03 | DK | Discard the singleton king on the losing club trick; retain both spades for partner's king and the final ace capture. |
| 04 | DK | Take the diamond lead with the king, then return the low diamond to partner's jack and ace. |

Input 03 keeps its four original synthetic hands with a canonical redeal:
West deals, South buys Sun and leads the opening club trick. Its recorded
auction, deck allocation and opening trick agree with that context; the
featured East hand, unique winning move and final project score are preserved.

Each new fixture supplies `caption` and `scenarioFamily` text. Selection carries
these into the render manifest and owner-approval document. Earlier fixtures
keep the existing caption and have no declared family. Family names describe
these sequences; they do not establish a different family from an actual
publication predecessor without reviewing that predecessor.

`build_notice_bundle()` and `validate_notice_bundle()` build and validate the
complete scene document without rendering. The existing design, which-card
opener, code box, fictional player-name draw, no-justification presentation and
render checks remain in force. Exact proof concerns the complete state; it is
not an actor-information-set guarantee.

For Saudi fictional names, pass the new public identity input through the
existing profile argument; the checked-in base profile remains unchanged:

```python
profile = notice.notice_profile(public=True)
profile["identity"] = json.loads(
    (notice.ROOT / "examples/synthetic-clip-identity.json").read_text(encoding="utf-8")
)
notice.validate_approved_v4_public_profile(profile)
bundle = notice.build_notice_bundle(
    rules, profile, fixture=selection["fixture"], golden=selection["golden"],
)
```

Retain the effective profile and its hash with the bundle. This explicit profile
input uses newly authored synthetic names; it restores no historical identity
configuration. Calling the CLI without this profile composition retains its
base name labels.

Fixtures and scene bundles are inputs, not released clips. Every final MP4 still
needs its own render evidence and explicit release. A catalog entry grants no
posting authority or permission to reuse an earlier clip.
