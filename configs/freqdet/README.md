# FreqDet experiment configs

This directory is reserved for experiments developed on branch
`freqdet/sfoc6-validation`.

Rules:

1. Do not modify detector backbone, neck, head, or detection losses.
2. Keep a paired baseline config for every dataset-detector protocol.
3. Name configs with the registered experiment ID from
   `../../../FreqDet_results/00_protocol/experiment_matrix.csv`.
4. Record the exact config path, commit, split hash, seed, and work directory in
   `../../../FreqDet_results/00_protocol/run_registry.csv`.
5. Keep ExDark FRBNet split, ExDark MAET split, and DarkFace 3:1:1 results in
   separate protocol groups.
6. New front-end implementations must share one code path with config switches
   for identity, F-only, S-only, and F+S ablations.
