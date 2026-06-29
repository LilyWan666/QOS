# Physical QPU Jobs

This directory describes the local QOS-Agent layout for IBM QPU physical
fidelity records copied from a QOS `ibm_quantum/jobs` data directory.

The records cover `ibm_marrakesh` and `ibm_torino` at utilization targets `30`, `60`, and `88`, with `8192` shots. Fidelity is stored in per-pair JSON files and uses `hellinger_mean`.

These files are physical QPU measurements. They are not local GPU simulation outputs and should not be used as strict Fig. 11 simulation provenance.
