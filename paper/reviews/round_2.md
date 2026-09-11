# Claude Review Round 2

- Model: `claude-fable-5`
- Session: `a2d3cb93-788b-4e88-ba11-21e72897328a`
- Scope: method reproducibility and no-lookahead validity

## Findings and decisions

1. **Seasonal period candidates are not specified.** Accepted. The appendix
   now lists the exact long-term, PEMS, and EPF period sets and states that
   periods are clipped at the context length and deduplicated.
2. **Residual shrinkage is underspecified.** Mostly rejected. The manuscript
   already defines the effective count, states that the moments are
   per-horizon and per-channel, and lists the `lambda` grid. The only missing
   constant was the numerical stabilizer, now stated as
   `epsilon = 1e-8`.
3. **Rolling bias may use partially realized residuals, and the perturbation
   check lacks a decision rule.** The leakage concern is rejected after
   checking the implementation: each residual component is written at its own
   target timestamp, and the cumulative statistics are queried at the strict
   prefix before the current origin. Horizon blocks group forecast leads and
   do not alter eligibility. The appendix now explains this mechanism and
   defines the overlap and rolling-bias parameters. No unverified numerical
   perturbation outcome was added.

## Verification

Tectonic rebuilt the manuscript successfully. The PDF remains 22 pages, and
the log contains no overfull boxes, undefined references, or undefined
citations.
