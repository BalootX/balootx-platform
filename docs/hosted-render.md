# GitHub-hosted rendering

`Hosted render` runs render-contract tests and two independent synthetic exports
on GitHub-hosted Ubuntu 24.04 runners, including draft and fork pull requests.
It uses read-only repository permissions, pinned actions, Node 22 and Python
3.13. No external credentials, private asset store or publishing service is used.
The existing CI lanes generate speech before consuming render assets too.

Run `python scripts/generate_public_render_audio.py` after installing eSpeak NG
1.51 and FFmpeg. It synthesizes 16 new utterances from public project text and
writes the compatible 15-clip and 16-clip packs. It records the generator binary
hash/version, voice, speed, exact input text, output hashes and measured audio
properties. The runtime uses eSpeak NG's built-in `ar` voice, with no MBROLA or
neural voice model. These are new synthesized outputs; they do not establish
rights, transcript accuracy or acoustic parity for any previous recording.
Generated WAVs are runtime outputs, never source-controlled.

Primary sources: [eSpeak NG 1.51 GPLv3 terms, section 2](https://github.com/espeak-ng/espeak-ng/blob/1.51/COPYING)
and [Arabic voice support and fully diacritized input limitation](https://github.com/espeak-ng/espeak-ng/blob/1.51/docs/languages.md).
The output provision depends on the output's content. This generator speaks the
project's own short callouts and explanation, and embeds no program source or
third-party recording. It makes no blanket claim about all synthesizer output.
The existing pack directory names are compatibility paths, not an accent or
smoothness certification. Auditory review remains necessary before releasing
these new voices as a product replacement.

The 904-byte card back is the retained repository-authored Majlis Sadu SVG,
under the root project LICENSE, with the existing manifest hash. It is present
in both asset trees. Standard Goodall cards retain their CC0 sidecars and Noto
fonts retain their existing notices. No historical media or private capture
receipts are distributed.

The existing geometry collector and render gate are preserved. The public policy
contains only the calibrated `namePills:1` / `buyerBadge:0` contact at hook,
question and reveal: maximum intersection width 169.578125 and height
5.8251953125 composition pixels. No other pair or phase is permitted. The policy
is bound to the current source tree; private calibration receipts are not inputs
to the public workflow. Geometry-only tests reproduce deal04's measured prompt /
buyer overflow (45.96875 by 35.8251953125), retaining FAIL / FAIL / PASS, and the
actual synthetic deal04 fixture remains barred before rendering by viewer/buyer
admission. Tests are not a replacement for historical capture evidence.

The retained design lock pins the unchanged approved component and styles plus
the fixed neutral identity configuration from the public import. It does not
claim pixel equivalence with historical media. Source/input, DOM, PNG, audio,
ffprobe, playback and final-file checks remain active. Generated artifacts use
the existing render gate and owner-approval contract; every MP4 remains
`postingAuthorized: false`. There is no operational switchover or deployment.

Artifacts retain the generated output and fresh speech manifests for seven days.
A successful hosted run qualifies that public candidate only. Posting still
requires the existing separate approval of the exact output file.
