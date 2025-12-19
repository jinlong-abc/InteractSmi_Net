# Environment
## Environment Setup

Create the environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate interactsmi_env
````

## Test the Environment

 Run the test command:

```bash
python -m test.Test_model_evalue > test_evalue_log.txt 2>&1
```

This will execute the test script and save logs to `test_evalue_log.txt`.

## Notes

* Python 3.9
* CUDA 12.4 required for PyTorch GPU support
* Main dependencies include:

  * `pytorch-cuda`
  * `torchvision`, `torchaudio`
  * `ipykernel`
  * `pymol-open-source`
  * `openbabel`
  * `rdkit`

Check the log file if there are any errors during execution.