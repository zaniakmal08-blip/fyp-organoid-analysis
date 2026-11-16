# Six-image retraining comparison

All six images were included in training. These are training-set results, not independent test accuracy.

Reference geometry: original saved manual ROIs, recalculated separately with a consistent provisional pilot calibration. Original CSV measurements were preserved; see ../../data/DATA_NOTES.md for the three new CSV issues.

| Image | Previous mask Dice | Updated mask Dice | Updated area error | Updated perimeter error | Updated circularity error |
|---|---:|---:|---:|---:|---:|
| B01_P01_A01_H2O2_000uM_D01_10x | 99.26% | 98.63% | -1.17% | +0.26% | -1.69% |
| B01_P01_A02_H2O2_12.5uM_D02_10x | 99.17% | 98.97% | +1.29% | +0.79% | -0.29% |
| B01_P01_A03_H2O2_025uM_D04_10x | 99.26% | 98.48% | +2.27% | +4.00% | -5.44% |
| B01_P01_A04_H2O2_050uM_D05_10x | 85.80% | 89.71% | -18.28% | +65.51% | -70.17% |
| B01_P01_A05_H2O2_100uM_D06_10x | 96.71% | 97.68% | -3.63% | -4.43% | +5.50% |
| B01_P01_B07_ROT_000nM_D01_10x | 97.84% | 98.87% | +0.52% | -0.54% | +1.61% |

## Average absolute percentage errors

| Parameter | Previous model | Updated model |
|---|---:|---:|
| Area | 5.334% | 4.529% |
| Perimeter | 7.043% | 12.589% |
| Circularity | 12.229% | 14.118% |
| AspectRatio | 0.907% | 1.573% |
| Roundness | 0.896% | 1.575% |
| Solidity | 2.312% | 2.566% |
| Feret | 4.603% | 4.511% |
| MinFeret | 2.615% | 0.763% |