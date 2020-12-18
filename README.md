# fpga_snn_models
A repository FPGA-friendly SNN models

## Requirements
- Python 3
- NumPy
- PyTorch
- BindsNET

## Scripts
### `snn_simulation.py`
A script to simulate the network to get the approximate integer weights to load onto the FPGA.

### `snn_weight_analysis.py`
A script to extract the weights from the network and convert them into 16 bit hex values.

### `snn_fpga_simulation.py`
A script to test the extracted weights with the original test data and evaluate the classification accuracy.