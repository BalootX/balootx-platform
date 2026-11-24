# BalootX design

## Rendered clips

- Use the table-dominant vertical Saudi Arabic composition. Lead with the card-choice question, keep the cards and table legible through the choice and reveal, and use fictional player identities. The [premium profile](configs/video/premium.yaml) governs presentation and export presets.
- Keep the majlis palette from the profile's `majlis` section. The red theme uses warm wood and rug tones with sand text and brass accents. The [Remotion styles](src/baloot_platform/video/remotion/src/style.css) apply the governed palette.
- Use Noto Sans Arabic for reading, Noto Kufi Arabic Bold for short display text, and monospace for the session code. Keep Arabic text right-to-left and the code left-to-right.
- The profile flag `majlis.research_notice` enables the one-line Arabic research notice in the red header band and the red session-code box. The [composition](src/baloot_platform/video/remotion/src/professional-prototype.tsx) uses separate graphics for audit media. Keep the on-screen question and the reveal clear of explanatory prose under the profile's presentation contract.
- Use the `vertical_social_v1` safe area from the [premium profile](configs/video/premium.yaml#L786) for essential text, cards and overlays. Leave the top and bottom overlay space and side gutters available. Decorative elements can extend beyond the content area.
- Keep the layout table pinned to its SHA-256 in the [layout gate](src/baloot_platform/video/remotion/scripts/layout-gate.mjs). The [approved table](src/baloot_platform/video/remotion/scripts/approved-layout-adjacencies.json) defines the notice height and the measured overlap allowance for each element pair and phase. Recalibrate and review changes to those allowances before adopting them. Measurement renders stay out of release.
- The side seat tags flow below their name pills, and buyer badges enclose the rotated purchase card. The layout table no longer allows the three west pill and buyer-badge overlaps. The south stack flows up from the hand and reserves room for a raised card on reveal.

## Foundry dashboard

- The [Foundry dashboard](src/baloot_platform/service/foundry_dashboard/index.html) uses an Arabic, right-to-left layout. The [Foundry styles](src/baloot_platform/service/foundry_dashboard/src/styles.css) use Noto Sans Arabic, dark green surfaces, gold accents and semantic colors for status. Follow the existing panel, form and table patterns.
- Keep scenario authoring, proof, execution and results easy to find. Keep status understandable in text and controls operable by keyboard. Preserve responsive layouts and reduced-motion support.
- The page budgets are p75 Largest Contentful Paint under 2.5 seconds and p75 Interaction to Next Paint under 200 ms.
