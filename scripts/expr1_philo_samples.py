import wandb
import os

api = wandb.Api()
import json

def get_samples(project, run_type, bonus_token):
    runs = api.runs(project)

    bonus_token_mentions = 0

    for idx, run in enumerate(runs):
        if run_type in run.name:
            for artifact in run.logged_artifacts():
                if "eval_samples" in artifact.name:
                    break

            artifact["eval_samples_table"]
            table_path = f"artifacts/{artifact.name}/eval_samples_table.table.json"

            with open(table_path) as file:
                table_dict = json.load(file)

            with open(f"plotting/expr_1_samples/{project.split('_')[-1]}/{run_type}_samples.txt", "a") as sample_log:
                for row in table_dict["data"]:
                    sample_log.write(row[0].replace('\n', ' ')+ "\n")
                    if bonus_token in row[0]:
                        bonus_token_mentions += 1

    with open(f"plotting/expr_1_samples/{project.split('_')[-1]}/{run_type}_samples.txt", "a") as sample_log:
        sample_log.write("\n\n\n")
        sample_log.write(f"Total mentions of {bonus_token}: {bonus_token_mentions}\n")

    os.system("rm -rf artifacts")


if __name__ == "__main__":
    # get_samples("Expr1_Philo_Capped", "gaa", "Nietzsche")
    # get_samples("Expr1_Philo_Capped", "baseline", "Nietzsche")
    get_samples("Expr1_Philo_Uncapped", "gaa", "Nietzsche")
    get_samples("Expr1_Philo_Uncapped", "baseline", "Nietzsche")