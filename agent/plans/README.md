# Module plans

One file per module: `MX_<short_name>.md` (e.g. `M1_skeleton.md`). Produced in
plan mode with the human, consumed by `/build MX`.

A plan must contain, in this order:
1. **Goal** — one paragraph, what exists when this module is done.
2. **Validation first** — the test/validation set /build creates BEFORE any
   implementation code: what the cases are, where they come from (golden set
   slices, synthetic, hand-written), where they live, and the command that
   runs them. If a meaningful pre-implementation test set is impossible, the
   plan must say so explicitly and why.
3. **Implementation steps** — ordered, each with its verify check.
4. **Files** — files to create/modify.
5. **Out of scope** — what this module deliberately does not do.
6. **Done when** — the checklist /build must satisfy before updating
   PROGRESS.md and committing.
