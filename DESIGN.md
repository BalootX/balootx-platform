# BalootX design

## Rendered clips

- Use the table-dominant vertical Saudi Arabic composition. Lead with the card-choice question, keep the cards and table legible through the choice and reveal, and use fictional player identities. The [premium profile](configs/video/premium.yaml) governs presentation and export presets.
- Keep the majlis palette from the profile's `majlis` section. The red theme uses warm wood and rug tones with sand text and brass accents. The [Remotion styles](src/baloot_platform/video/remotion/src/style.css) apply the governed palette.
- Use Noto Sans Arabic for reading, Noto Kufi Arabic Bold for short display text, and monospace for the session code. Keep Arabic text right-to-left and the code left-to-right.
- The profile flag `majlis.research_notice` enables the one-line Arabic research notice in the red header band and the red session-code box. The [composition](src/baloot_platform/video/remotion/src/professional-prototype.tsx) uses separate graphics for audit media. Keep the on-screen question and the reveal clear of explanatory prose under the profile's presentation contract.
- The [research-notice renderer](scripts/render_research_notice.py) uses the [approved public v4 design binding](src/baloot_platform/video/render_bundle.py). See [Manual render kickoff](DELIVERY.md#manual-render-kickoff) for the build and kickoff receipts and the worker launch command.
- Use the `vertical_social_v1` safe area from the [premium profile](configs/video/premium.yaml#L786) for essential text, cards and overlays. Leave the top and bottom overlay space and side gutters available. Decorative elements can extend beyond the content area.
- The [approved layout pin](src/baloot_platform/video/remotion/scripts/approved-layout-pin.json) supplies the exact table SHA-256 to the JavaScript layout gate and Python media and release validation. The [approved table](src/baloot_platform/video/remotion/scripts/approved-layout-adjacencies.json) defines the notice height and the measured overlap allowance for each element pair and phase. Recalibrate and review changes before updating the table and its pin together. Measurement renders stay out of release.

## Foundry dashboard

- The [Foundry dashboard](src/baloot_platform/service/foundry_dashboard/index.html) uses an Arabic, right-to-left layout. The [Foundry styles](src/baloot_platform/service/foundry_dashboard/src/styles.css) use Noto Sans Arabic, dark green surfaces, gold accents and semantic colors for status. Follow the existing panel, form and table patterns.
- Keep scenario authoring, proof, execution and results easy to find. Keep status understandable in text and controls operable by keyboard. Preserve responsive layouts and reduced-motion support.
- On the overview, keep readiness details disabled while their data loads and the layout is still settling. Once enabled, each detail opens by touch, mouse or keyboard.
- The page budgets are p75 Largest Contentful Paint under 2.5 seconds and p75 Interaction to Next Paint under 200 ms.

## Browser requests

- The served pages and the Foundry dashboard sources call the API through the [shared request deadlines](src/baloot_platform/service/static/shared/request.js). `/foundry` serves a committed bundle that is not yet built from those sources, so the served dashboard keeps its older request handling until [#244](https://github.com/BalootX/balootx-platform/issues/244) is fixed. Interactive calls use the request deadline. Synchronous Research calls that run a solver or prepare a candidate use the long-call deadline, which exceeds the longest covered server budget by a stated margin. A request that reaches its deadline ends once, with no retry. It shows the profile's `request_timeout` message, which asks the person to refresh and check the result before trying again.
- The served pages load the request module through the [request loader](src/baloot_platform/service/static/shared/request-loader.js). A failed load shows the profile's `request_unavailable` message, which asks the person to try again and to refresh if it persists. The next request loads the module again.

## Served pages

- The `/app`, `/play` and `/c` pages take their visible copy from `challenge_copy` in the [premium profile](configs/video/premium.yaml). The page templates and scripts hold copy keys only. Keep the Arabic pages right-to-left, and mark room codes, game numbers and case numbers left-to-right.
- `/app` requires the full set of app copy keys declared in the [video schema](schemas/profiles/video.schema.json). The guest display name follows the identity service's length limit. Exact Research revisions keep the video payload they were saved with.
