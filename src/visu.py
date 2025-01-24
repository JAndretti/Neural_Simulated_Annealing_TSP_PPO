import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from model import TSPActor
from problem import TSP
from sa import sa

from HP import _HP, get_script_arguments

cfg = _HP("src/test.yaml")
cfg.update(get_script_arguments(cfg.keys()))


# Function to set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Function to extract the loss value from a filename
def extract_loss(filename):
    try:
        # Identify and extract the "_loss_X" portion
        return float((filename.split("_")[-1])[:-3])
    except ValueError:
        pass
    return float("inf")  # Return a large number if extraction fails


# Function to load the model weights from wandb
def load_model(model, folder):
    # List all files in the directory
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    if files:
        # Find the file with the smallest loss
        file_with_min_loss = min(files, key=extract_loss)
    model.load_state_dict(torch.load(folder + file_with_min_loss, weights_only=True))
    print(f"Loaded model from {file_with_min_loss}")
    return model


def main():
    # Initialize the actor model
    actor = TSPActor(cfg["EMBEDDING_DIM"], device=cfg["DEVICE"])
    if cfg["LOAD_MODEL"]:
        actor = load_model(actor, cfg["MODEL_DIR"])
    set_seed(cfg["SEED"])
    problem = TSP(cfg["VISU_DIM"], 1, device=cfg["DEVICE"])
    params = problem.generate_params()
    params = {k: v.to(cfg["DEVICE"]) for k, v in params.items()}
    problem.set_params(**params)
    # Find initial solutions
    init_x = problem.generate_init_x()

    alpha = np.log(cfg["STOP_TEMP"]) - np.log(cfg["INIT_TEMP"])
    cfg["ALPHA"] = np.exp(alpha / cfg["OUTER_STEPS"]).item()
    with torch.no_grad():
        actor.eval()
        train = sa(
            actor,
            problem,
            init_x,
            cfg,
            replay=None,
            baseline=False,
            greedy=False,
            record_state=True,
        )
    logits = train[
        "distributions"
    ]  # list of distributions of size VISU_STEPS (dist actor1, dist actor2)
    states = train[
        "states"
    ]  # list of states of size VISU_STEPS (node_nbr, coord1, coord2, temp)
    actions = train["actions"]  # list of actions of size VISU_STEPS (actor1, actor2)
    acceptance = train["acceptance"]  # list of bool of size VISU_STEPS
    costs = train["costs"]  # list of costs of size VISU_STEPS
    # min_cost = train["min_cost"]
    iter = 0
    for logit, state, action, acc, cost in zip(
        logits, states, actions, acceptance, costs
    ):
        if iter == cfg["OUTER_STEPS"] - 1:
            break
        fig, (ax1, ax2) = plt.subplots(2, figsize=(25, 15))

        prob1 = torch.softmax(logit[0], dim=-1)
        prob2 = torch.softmax(logit[1], dim=-1)

        # Premier graphique en bar pour logit[0]
        ax1.bar(range(len(prob1.squeeze())), prob1.squeeze())
        ax1.set_title("Distribution prob actor 1")

        # Second graphique en bar pour logit[1]
        ax2.bar(range(len(prob2.squeeze())), prob2.squeeze())
        ax2.set_title("Distribution prob actor 2")

        fig.suptitle(f"{action}")

        plt.show()


if __name__ == "__main__":
    main()
