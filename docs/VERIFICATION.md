# Repository verification — 16 November 2026

- Dependency-free structure, Python syntax, Markdown links and checkpoint checksum: passed.
- Local pilot import/checksum verification: 38 files passed.
- CNN inspector: 75 pixel calculations across 25 operations passed; largest absolute difference approximately 4.16e-6.
- Fiji integration: original three manual-reference comparisons, predicted ROI/CSV/ZIP exports and original image dimensions passed.
- Calibration scaling: doubling scale multiplied area by four and length by two; dimensionless values unchanged.
- Invalid-input handling: blank TIFF, multi-plane stack and PNG rejected.

Machine-readable reports are in `results/verification-2026-11-16`. Packages are recorded in `requirements-tested.txt`. The checks ran with the existing local environment and Fiji; a fresh dependency installation has not been tested. These are software checks, not independent segmentation accuracy or lab calibration validation.
