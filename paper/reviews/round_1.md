# Claude Review Round 1

- Model: `claude-fable-5`
- Session: `5d43ac16-5d57-469b-887a-004bfd2cfeb1`
- Scope: introduction, method, experimental design, results, and conclusion

## Findings and decisions

1. **A100 headline and holdout counts appear arithmetically inconsistent.**
   Accepted as a labeling defect. The 562/585 result is the A100 primary
   replication, while 437/462 is the original A10G frozen confirmation. The
   manuscript now states that the runs have different hardware identities and
   are not an additive partition.
2. **The conservative selector points to the wrong specification.**
   Accepted. The exact margin and family order are in Appendix
   `app:search`, not the main evaluation-protocol section.
3. **The abstract over-attributes portfolio evidence to error memory.**
   Partly accepted. The title and method contribution remain because error
   memory is the central new retrieval object and the additional-systems study
   selects it in most cells. The abstract conclusion now distinguishes broad
   evidence for post-hoc revision from direct evidence for historical error
   memory beyond individual neural backbones.
4. **Cell-wise reductions and ratios of displayed macro-averages look
   contradictory.** Accepted as a presentation issue, not a numerical error.
   The text now explains that the macro-average ratio weights by baseline
   error, while paired reductions weight cells equally.

## Verification

Tectonic rebuilt the manuscript successfully. The PDF remains 22 pages, the
main text still ends on page 9, and the log contains no overfull boxes,
undefined references, or undefined citations.
