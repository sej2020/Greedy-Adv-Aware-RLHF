import pandas as pd
import wandb

api = wandb.Api()

def retrieve_metric(project, run_type):
    runs = api.runs(project)
    
    kl_df = pd.DataFrame()
    ent_df = pd.DataFrame()
    reward_df = pd.DataFrame()
    eval_reward_df = pd.DataFrame()

    for idx, run in enumerate(runs):
        if run_type in run.name:
            print('.', end='', flush=True)
            hist = run.scan_history(keys=['kl_penalty', 'entropy_bonus', 'mean_reward'])
            kl_df[run.name] = [row['kl_penalty'] for row in hist]
            ent_df[run.name] = [row['entropy_bonus'] for row in hist]
            reward_df[run.name] = [row['mean_reward'] for row in hist]

            eval_reward = run.summary['mean_eval_reward']
            eval_reward_df[run.name] = [eval_reward]
        
    kl_df.to_csv(f'plotting/expr1_result_csv/{project.split('_')[-1]}/kl_{project}_{run_type}.csv')
    ent_df.to_csv(f'plotting/expr1_result_csv/{project.split('_')[-1]}/ent_{project}_{run_type}.csv')
    reward_df.to_csv(f'plotting/expr1_result_csv/{project.split('_')[-1]}/reward_{project}_{run_type}.csv')
    eval_reward_df.to_csv(f'plotting/expr1_result_csv/{project.split('_')[-1]}/eval_reward_{project}_{run_type}.csv')


if __name__ == "__main__":
    retrieve_metric("Expr1_Philo_Uncapped", "gaa")
    retrieve_metric("Expr1_Philo_Uncapped", "baseline")