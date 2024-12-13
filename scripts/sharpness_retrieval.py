import json
import wandb
import matplotlib.pyplot as plt
import os

api = wandb.Api()

def retrieve_metric(project):
    runs = api.runs(project)
    
    exp_data = {}
    unexp_data = {}
    for idx, run in enumerate(runs):
        print('.', end = '', flush=True)
        hist = run.scan_history(keys = ["loss_list_full", "loss_list_unembed", "loss_list_mlp"])
        for i in hist:
            if i["loss_list_full"] is not None:
                if "Unexp" in run.name:
                    unexp_data[run.name] = i
                else:
                    exp_data[run.name] = i

    os.makedirs("plotting/sharpness", exist_ok=True)
    json.dump(exp_data, open("plotting/sharpness/exp_data.json", "w"))
    json.dump(unexp_data, open("plotting/sharpness/unexp_data.json", "w"))


if __name__ == "__main__":
    retrieve_metric("Sharpness")