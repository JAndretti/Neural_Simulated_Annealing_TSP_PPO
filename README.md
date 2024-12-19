# Neural Simulated Annealing for TSP

This project implements a Neural Simulated Annealing (NSA) approach for solving the Traveling Salesman Problem (TSP). The NSA method combines neural networks with simulated annealing to find optimal or near-optimal solutions for TSP instances.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation


1. Create and activate a virtual environment:
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Configuration

The configuration for the project is specified in the [test.yaml](http://_vscodecontentref_/0) file. You can modify this file to change the parameters for the model, problem, and training process.

### Training

To train the model, run the following command:
```sh
python src/main.py
```
Make sure to use the right config, in *HP.yaml*
### Evaluation

To evaluate the model, run the following command:
```sh
python src/eval.py
```
Make sure to use the right config, in *test.yaml*
### Results

The results of the evaluation will be saved in the models directory. You can view the results by opening the generated text files.