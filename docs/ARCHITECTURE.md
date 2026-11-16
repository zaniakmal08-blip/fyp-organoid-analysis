# Architecture

`train_three.py` defines TinyUNet (29,321 parameters) and the legacy three-image baseline. `train_prototype.py` uses that model with percentile normalisation, letterboxing and boundary-weighted fitting. `organoid_core.py` owns TIFF validation, model loading, inverse resize, polygon extraction and ImageJ measurement. `imagej_helper/MeasureOrganoid.java` runs Fiji's ImageJ API in a headless Java process. `prototype_app.py` serves the local upload UI; `network_visualizer.py` exposes model operations and calculations.

Keep HTML files next to their Python servers. `project_paths.py` centralises data, model, result and Fiji locations. Local results are ignored by Git. Both servers bind to localhost and are research tools, not hosted multi-user services.

Calibration changes physical measurements and must be verified for each acquisition setting. A 2D mask does not directly measure 3D volume, neuron fraction, viability or differentiation.
