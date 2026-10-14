# Early Step 8 foundation — ATC system evaluation

The paired ATC sensing/planning evaluation was implemented before the project
roadmap was reconciled. It is retained as a Step 8 foundation, not as Step 6.

It produces 10,000-sample timestamp-block bootstrap intervals for 654 sensing
timestamps, 396 paired path comparisons, and 316 replanning events. Run it with
`make atc-evaluation`; outputs are written to
`experiments/outputs/atc-step8-foundation` and paper copies to
`paper/pacificvis/figures/generated/atc-step8-foundation`.
