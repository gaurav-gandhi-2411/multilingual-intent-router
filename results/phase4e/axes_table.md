## Multi-axis selection axes: candidate - v1 (CI of the improvement; reductions for flip rate / ECE)

v1 (e*=9): F1 0.9336, AUROC 0.8799, rej@95 0.4201, flip 0.2381, agreement 0.8534, ECE 0.0263

| candidate | e* | (a) CV F1 | (b) dAUROC | (b) dRej@95 | (c) flip | (d) agreement | (e) ECE | eligible | improved axes |
|---|---|---|---|---|---|---|---|---|---|
| a3 | 6 | +0.0130 [+0.0045, +0.0220] ok, improved | +0.0001 [-0.0123, +0.0127] ok | +0.0020 [-0.0468, +0.0495] ok | +0.1270 [-0.2143, -0.0437] INELIGIBLE | +0.0555 [+0.0395, +0.0721] ok, improved | -0.0013 [-0.0070, +0.0133] ok | False | a,d |
| a1 | 11 | -0.0109 [-0.0206, -0.0014] INELIGIBLE | -0.0136 [-0.0276, -0.0001] INELIGIBLE | -0.0529 [-0.0871, -0.0187] INELIGIBLE | -0.2024 [+0.1389, +0.2698] ok, improved | -0.0200 [-0.0358, -0.0048] INELIGIBLE | +0.0114 [-0.0238, +0.0001] ok | False | c |
| a1a3 | 6 | +0.0002 [-0.0103, +0.0111] ok | -0.0020 [-0.0177, +0.0134] ok | -0.0185 [-0.0667, +0.0309] ok | -0.1865 [+0.1230, +0.2500] ok, improved | +0.0463 [+0.0272, +0.0658] ok, improved | -0.0003 [-0.0111, +0.0116] ok | True | c,d |
