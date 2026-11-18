# Selected model

**RandomForest** ranked first. Selection gives equal weight to grouped validation macro F1 and student-test observation top-1 accuracy. Ties use lower student-test candidate log loss, then lower batch inference time, then model name.

| Model | CV macro F1 | Student top-1 | Sum | Log loss | Inference (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.887226 | 0.859760 | 1.746986 | 0.071286 | 0.467143 |
| MLP | 0.880906 | 0.848069 | 1.728975 | 0.080066 | 0.041157 |
