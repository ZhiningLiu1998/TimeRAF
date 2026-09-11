# Core-Insight Narrative Review

- Source: initiation slides, including the event/residual retrieval sequence
- Reference structure: TimeFuse introduction and method narrative
- Scope: motivation, claim boundaries, Figure 1, and final manuscript layout

## Narrative decision

The paper now starts from the distinction between regular structure and
model-specific blind spots. Whole-window similarity can be dominated by
trend and seasonality that the forecaster already captures, while an analog
future can import an old level or phase. TimeRaf therefore preserves the
frozen base and retrieves only a historical residual.

The exposition follows the TimeFuse progression from a fine-grained
observation to a research question, two method components, practical
advantages, and layered empirical evidence. The manuscript does not cite or
repeatedly name TimeFuse outside its normal related-work and benchmark roles.

## Claim boundary

Residual memory is presented as a model-relative decomposition, not as a
predefined high-frequency event representation. The current descriptor still
summarizes the full context and base forecast. It does not implement an
explicit rarity detector, event-only memory, no-event gate, or
component-specific mixture of retrievers; these remain future work.

No experimental design, result paragraph, table, or reported number changed
in this revision.

## Verification

- Regenerated all paper figures with the project virtual environment.
- Rechecked the Quip design reference; Figure 1 retains its shallow
  left-to-right flow, bounded stages, direct labels, and restrained accents.
- Visually inspected Figure 1 alone and in the compiled first page.
- Compiled the manuscript with the paper-local Tectonic binary.
- Confirmed 22 total pages, Figure 1 on page 1, Figure 2 on page 3, and main
  text ending on page 9.
- Confirmed no overfull boxes, undefined references, or undefined citations.
- Visually inspected the final main-text page for clipping and overlap.
