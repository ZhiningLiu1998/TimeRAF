# Figure 1 Flaticon Icon Debate

- Reviewer: Claude Opus through ASBX Claude Code 2.1.246.731
- Scope: Flaticon asset semantics, Figure 1 standalone and manuscript-scale
  rendering, attribution, and transparent-PNG embedding
- Constraint: preserve the stadium-traffic example and retrieval architecture;
  use sourced Flaticon assets rather than hand-drawn substitutes
- Process: two claim-response rounds, local visual inspection, and local
  adjudication of every comment

Two ASBX attempts to attach the rendered PNGs directly stalled without a
review response. The completed Claude rounds therefore used a dimensioned
audit packet derived from the standalone and compiled-page renders. The local
review independently inspected both rendered artifacts before and after each
fix.

## Asset Decision

Figure 1 uses three original PNGs from Flaticon's official CDN:

1. Stadium 53213 by Freepik for the event-night setting;
2. Magnifying Glass 49116 by Magnific for whole-window search; and
3. Database 51319 by Freepik for residual memory.

The untouched source bytes, detail and CDN URLs, authors, license links,
download date, transformations, pixel dimensions, effective DPI, and SHA-256
values are recorded in `paper/assets/flaticon/ATTRIBUTION.md`.

## Round 1

Claude made eleven claims.

1. **Small annotation text:** partly accepted. The 7.4--8.6 source-unit text
   was the main page-scale risk. Headers were increased from 10.8 to 11.2,
   annotations to at least 8.4, and the other small labels to 8.8--9.4. A
   blanket 11-unit minimum was rejected because it would crowd the dense plot.
2. **Ambiguous pixel-scale evidence:** accepted. Pixel dimensions were
   replaced with source dimensions, alpha-crop dimensions, and final effective
   DPI.
3. **Red and green icons duplicate verdict colors:** rejected. These colors
   consistently identify the failure and success branches in the title,
   arrows, curves, backgrounds, and verdicts. Shape and text make the encoding
   redundant.
4. **Panel geometry might be misaligned:** rejected. The actual coordinates
   align the two right-panel icon and text columns, and the top titles differ
   by only one source unit; the latter was revisited in Round 2.
5. **Add `baseline` and `ours` to the headers:** rejected. The depicted
   raw-future and residual operations plus the caption already establish the
   comparison; ownership labels would add clutter and overstate the claim.
6. **Record source resolution and interpolation:** accepted. The provenance
   now proves 1603--2048 effective DPI with no upsampling, and the RGB image
   XObjects set `/Interpolate true`.
7. **Normalize icon optical mass:** rejected after inspecting the page render.
   The three marks have comparable visual height, and each remains distinct.
8. **Standardize icon and panel-letter order:** rejected as already satisfied.
   Every header uses icon, panel letter, then title.
9. **Red/green accessibility:** rejected as a defect. Glyph, text, line style,
   cross/check, and vertical position all survive grayscale or hue loss.
10. **Soft-mask PDF conformance:** rejected. PDF 1.4 supports `/SMask`; the
    implementation uses DeviceRGB images with DeviceGray masks, no white
    matte, and no embedded ICC profile.
11. **Attribution detail:** no defect found. The caption named both authors and
    Flaticon, while the repository retained per-icon provenance.

## Round 2

Claude challenged the Round 1 decisions and narrowed the remaining questions
to four concrete points.

1. **Printed text remains small:** partly accepted. The smallest remaining
   labels were raised again so all Figure 1 text is at least 9.0 source units.
   The final compiled-page inspection found no clipping or collision.
2. **Title baselines differ by one source unit:** accepted. Panels (a) and (b)
   now share `y=198`; panel (c) uses the same relative baseline in the lower
   band (`y=93`, exactly 105 units below).
3. **Icon tint is valid only if the icons are branch-specific:** the condition
   is satisfied. The magnifier appears only in the whole-window/raw-copy
   branch, and the database appears only in the residual-memory branch.
4. **Printed attribution should expose the site reference:** accepted
   conservatively. Although the existing `Flaticon` text was already a
   hyperlink and matched the non-web attribution wording, the caption now
   visibly includes `www.flaticon.com`.

Claude upheld the decisions not to add ownership labels, neutralize the branch
icons, replace the Flaticon assets, or change the alpha-mask implementation.

## Final Decision

The final figure retains the scientific data and architecture while replacing
all three hand-drawn semantic marks with sourced Flaticon assets. The icon
roles are branch-specific, the cross/check remain the only verdict symbols,
and the visible caption attribution covers both authors and Flaticon. No
Claude comment remained both evidence-backed and unaddressed.
