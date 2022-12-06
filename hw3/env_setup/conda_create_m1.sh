# conda remove -n "dl33"
conda env create -n "dl35" -f csci1470-m1.yml

## Install new environment.
python -m ipykernel install --user --name dl335 --display-name "dl35"
conda activate dl35

## Tensorflow metal might malfunction for some students. Better to remove it
pip uninstall tensorflow-metal
