# Selected pilot model

Source: original `training_pilot/prototype_model/tiny_unet.pt`.

SHA-256: `f35b680a14671ea23703c7784be8d5bf8366e1262d797e9906b86de449fe5b44`.

TinyUNet, 29,321 parameters; single grayscale input, binary segmentation. The selected checkpoint uses aspect-preserving percentile preprocessing. Six images were used in fitting. `training_metrics.json` contains fitting scores; these differ from the exported polygon-mask comparison in `experiments/six-image-pilot`.

Intended use: engineering prototype and supervised boundary review. No independent validation or biological decision-making performance is established. A04 remains a major failure case. Do not use the output as a calibrated reference without verified scale and expert boundary review.

The checkpoint is small enough for regular Git. Keep new candidates under ignored results; promote a reviewed checkpoint with its hash, data version and experiment entry. Historic selection used 300 epochs, learning rate 0.001 and Dice weight 3 according to the saved six-image update notes. Re-running from this checkpoint continues fitting rather than reconstructing its original initial state.
