import os
import subprocess
import glob
import argparse
import pandas as pd
from summary_plots import plot_fov_dur_sac_amp_hists, plot_sac_ang_hists, plot_ior_stats, plot_object_eval, filter_dfs_for_eval_figures

import scanpath_utils as su


# start this in a tmux window, will run all folders in parallel using srun
# don't forget: conda activate ...
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str)
    # parser.add_argument("--eval_only_3sec", type=bool, default=False)
    parser.add_argument('--eval_only_3sec', default=False, action='store_true')
    args = parser.parse_args()
    RES_DIR = args.experiment_dir
    EVAL_ONLY_3SEC = args.eval_only_3sec
    config_name = RES_DIR.split("/")[-1][19:]
    print(f"Experiment {RES_DIR}, config: {config_name}")
    
    res_files = []
    for path, subdirs, files in os.walk(RES_DIR):
        for name in files:
            if name == "raw_results.pickle4":
                res_files.append(os.path.join(path, name))

    print("#Trials in simulation run: ", len(res_files))
    if "_TEST_" in RES_DIR:
        print(f"Using test data for evaluating {RES_DIR}...")
        data_path = "/scratch/vito/scanpathes/scanpath_data/test_data"
    else:
        print(f"Using training data for evaluating {RES_DIR}...")
        data_path = "/scratch/vito/scanpathes/scanpath_data/training_data"
    df = pd.concat([su.evaluate_model_trial(res, data_path=data_path) for res in res_files], ignore_index=True)
    df.to_csv(os.path.join(RES_DIR, 'df_res_fov.csv'))

    # Load ground truth foveation dataframe
    # df_gt = pd.read_csv("/home/vito/Documents/eye_data_EM-UnEye_2023-11-29/df_res_gt_fov_train.csv")
    df_gt = pd.read_csv("/scratch/vito/scanpathes/scanpath_data/df_res_gt_fov_all.csv")
    
    df, df_gt = filter_dfs_for_eval_figures(df, df_gt, EVAL_ONLY_3SEC, RES_DIR)


    # PLOTTING
    plot_fov_dur_sac_amp_hists(df, ground_truth=df_gt, savedir=RES_DIR, name_flag=config_name)
    plot_sac_ang_hists(df, ground_truth=df_gt, savedir=RES_DIR, name_flag=config_name)
    plot_ior_stats(df, ground_truth=df_gt, savedir=RES_DIR, name_flag=config_name)
    plot_object_eval(df, ground_truth=df_gt, savedir=RES_DIR, name_flag=config_name)
    # plot_fov_dur_sac_amp_hists(df, savedir=RES_DIR, name_flag=config_name)
    # plot_sac_ang_hists(df, savedir=RES_DIR, name_flag=config_name)
    # plot_ior_stats(df, savedir=RES_DIR, name_flag=config_name)
    # plot_object_eval(df, savedir=RES_DIR, name_flag=config_name)

    # # Run the frame-wise visualizations
    # experiments = glob.glob(os.path.join(RES_DIR, "config_0", "*"))
    # for p in experiments:
    #     dp.process_experiment_data(p, dpi=100)


if __name__ == "__main__":
    main()
