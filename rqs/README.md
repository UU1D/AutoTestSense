# Paper Experiment Artifacts

This directory organizes the artifacts and entry points used by the paper's
research questions. Library construction itself is documented in the package
[README](../README.md).

| Directory | Scope | Released material |
|---|---|---|
| [`rq1/`](rq1/) | Human assessment of the mined knowledge | Sample index for 302 generalized rules and 848 source instances |
| [`rq2/`](rq2/) | Effect of generalized commonsense on GUI bug detection | Three baselines, two detector models, with/without injection, 1,812 parsed outputs |
| [`rq4/`](rq4/) | Instance-level commonsense ablation | Three baselines on Gemini, 453 parsed outputs |

RQ2 uses the generalized commonsense library. RQ4 changes only the knowledge
representation used for retrieval and injection: it retrieves directly from
the 3,708 extracted instance-level items. The two experiment modules share
retrieval and detector utilities but keep their inputs and outputs separate.
