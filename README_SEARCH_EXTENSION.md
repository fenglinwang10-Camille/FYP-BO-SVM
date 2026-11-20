# Training-only continuation from 50 to 100 evaluations

This exploratory budget study continues the completed uniform BO and RS runs
from experiment 13 (score-based recording vote tie-breaking). It does not
reinitialise either method, reread outer-test signals or evaluate outer-test
performance. Both methods receive fifty additional candidate evaluations in
each of the same five outer training sets: 500 new inner-CV evaluations total.

`extend_nested_search.py` verifies the original protocol, source hashes,
package versions, complete candidate numbering, GP observation coordinates and
negative-accuracy targets. It forks the saved checkpoint into a **new** output
directory, preserving the first 50 histories, GP, optimiser RNG and next-point
state. The new protocol records the original protocol/checkpoint digests. The
RS stream is checked to preserve its first fifty random configurations. The
sixteen initial candidates remain shared; no additional random initialization
is inserted. Existing source experiment files and results are untouched.

The first new candidate is evaluation 51. Each subsequent checkpoint is saved
with the same original run-lock/atomic-write machinery. Tests verify that only
new evaluations run, the source checkpoint is unchanged, and an interrupted
continuation gives the same points and scores as an uninterrupted continuation.

Run `run_search_extension.ps1` to start hidden in the background or resume.
Outputs are in
`results/14_training_search_extension_100/split_42_inner_1729/search_42`.
See each fold's `progress.json`, the timestamped log and root
`training_search_summary.csv` for scores at evaluation 50, current best score,
improvement after 50 and last improvement number. `summary.json` is generated
only after all ten method/fold histories contain 100 evaluations.

This is an inner-validation budget sensitivity analysis prompted by inspection
of the original fifty-evaluation histories. It must not be presented as a
previously unplanned independent confirmation, or as an outer-test performance
improvement. A flat tail does not prove convergence to an optimum. The final
training-selected parameters are saved, but this program never evaluates them
on the previously inspected outer test folds.

The default output root includes `100`; if choosing a different budget, also
provide a distinct `--output-dir` to preserve the fixed existing protocol.
