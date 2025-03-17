#!/bin/bash
# run_all_sweeps.sh

echo "Running parameter sweeps with 6 workers..."

# Run classic sweeps
# python run_experiment.py k-dt-original --max-workers 6
# python run_experiment.py k-dt-base --max-workers 6
# python run_experiment.py k-dt-high-carrying --max-workers 6
# python run_experiment.py k-dt-ones --max-workers 6
# python run_experiment.py k-dt-ones-low-carrying --max-workers 6
# python run_experiment.py k-dt-unit-start-low-carrying --max-workers 6
# python run_experiment.py k-dt-unit-start-high-carrying --max-workers 6

echo "All parameter sweeps complete!"

echo "Copying sweep charts to ./public/sweeps..."

# Copy the sweep charts to./public/sweeps
# cp -r ./data/results/sweep_visualization.png ./public/adaptive-sweeps/k-dt-original.png
# cp -r ./data/results/sweep_visualization.png ./public/adaptive-sweeps/k-dt-base.png
# cp -r ./data/results/sweep_visualization.png ./public/adaptive-sweeps/k-dt-high-carrying.png
# cp -r ./data/results/sweep_visualization.png ./public/adaptive-sweeps/k-dt-ones.png
# cp -r ./data/results/sweep_visualization.png ./public/adaptive-sweeps/k-dt-ones-low-carrying.png
# cp -r ./data/results/sweep_visualization.png ./public/adaptive-sweeps/k-dt-unit-start-low-carrying.png
# cp -r ./data/results/sweep_visualization.png ./public/adaptive-sweeps/k-dt-unit-start-high-carrying.png

echo "Sweep charts copied to ./public/adaptive-sweeps!"