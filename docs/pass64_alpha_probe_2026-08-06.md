# Pass 64 alpha probe on cached CUB embeddings

Following the blind proposer’s cheap diagnostic, I fit the class-mean
`C-1=99`-dimensional subspace on the training embeddings only, decomposed each
test vector into parallel and orthogonal components, and swept
`normalize(P_parallel z + alpha P_perp z)`. This is post-hoc evidence, not a
training method or a claim.

| alpha | R@1 | local failures | between failures |
|---:|---:|---:|---:|
| 0 | .6555 | 901 | 1140 |
| .25 | .6580 | 888 | 1138 |
| .50 | .6653 | 895 | 1088 |
| .75 | .6757 | 905 | 1016 |
| 1.00 | .6816 | 938 | 948 |
| 1.25 | .6813 | 974 | 914 |
| 1.50 | .6825 | 1003 | 878 |
| 2.00 | .6789 | 1067 | 835 |
| 3.00 | .6673 | 1154 | 817 |

The best alpha is about 1.5, only +0.085 points over alpha=1 in this centered
probe, and far below the proposer’s 1.5-point reopening threshold. Local and
between failures trade monotonically. This weakens the rank-preservation lane;
it does not prove the broad closure lemma. No GPU run followed.
