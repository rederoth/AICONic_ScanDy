import glob
import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import collections
import yaml
import matplotlib.image as mpimg
from scipy import stats
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

from summary_plots import evaluate_all_obj, get_BDIR_per_frames, fovdur_vs_sacang


label_dict = {"dv": r"DDM threshold $\theta_{DDM}$",
              "sig" : r"DDM noise level $\sigma_{DDM}$",
              "sens" : 'Size of Gaussian sensitivity [dva]',
              "task" : "Added number: Saliency",
              "entropy" : "Added number: Uncertainty"
             }


def fix_hist_step_vertical_line_at_end(ax):
    """
    Get rid of vertical lines on the right of the histograms, as proposed here:
    https://stackoverflow.com/questions/39728723/vertical-line-at-the-end-of-a-cdf-histogram-using-matplotlib

    :param ax: Axis to be fixed
    :type ax: matplotlib.axes._subplots.AxesSubplot
    """
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])

def plot_cdf_fd_sa(df_test_res, df_train_res, df_gt, df_other=None, other_name=""):
    df_test_gt = df_gt[df_gt["video"].isin(df_test_res["video"].unique())]
    df_train_gt = df_gt[df_gt["video"].isin(df_train_res["video"].unique())]
    fig, axs = plt.subplots(1,2,dpi=150, figsize=(9.5,3), sharey=True) # fov_dur & sac_amp 
    for modus in ["train", "test"]:
        if modus == "train":
            gt_amp_dva = df_train_gt["sac_amp_dva"].dropna().values
            gt_dur_ms = df_train_gt["duration_ms"].dropna().values
            res_amp_dva = df_train_res["sac_amp_dva"].dropna().values
            res_dur_ms = df_train_res["duration_ms"].dropna().values
            kwargs = {"alpha":0.3, "lw":2} #"ls":"dotted", "lw":3,         
        else:
            gt_amp_dva = df_test_gt["sac_amp_dva"].dropna().values
            gt_dur_ms = df_test_gt["duration_ms"].dropna().values
            res_amp_dva = df_test_res["sac_amp_dva"].dropna().values
            res_dur_ms = df_test_res["duration_ms"].dropna().values
            kwargs = {"alpha":1, "lw":3}
        nbins = 60  # np.linspace(1, 4, 100)  # 60
        axs[0].hist(np.log10(gt_dur_ms), nbins, density=True, histtype='step', cumulative=True, label=f"Humans {modus}", color="xkcd:red", **kwargs)
        axs[0].hist(np.log10(res_dur_ms), nbins, density=True, histtype='step', cumulative=True, label=f"Model {modus}", color="xkcd:blue", **kwargs)
        
        axs[1].hist(gt_amp_dva, nbins, density=True, histtype='step', cumulative=True, label=f"Humans {modus}", color="xkcd:red", **kwargs)
        axs[1].hist(res_amp_dva, nbins, density=True, histtype='step', cumulative=True, label=f"Model {modus}", color="xkcd:blue", **kwargs)

    if df_other is not None:
        axs[0].hist(np.log10(df_other["duration_ms"].dropna().values), nbins, density=True, histtype='step', cumulative=True, label=other_name, color="xkcd:teal", alpha=1, lw=2)
        axs[1].hist(df_other["sac_amp_dva"].dropna().values, nbins, density=True, histtype='step', cumulative=True, label=other_name, color="xkcd:teal", alpha=1, lw=2)
        
    fix_hist_step_vertical_line_at_end(axs[0])
    fix_hist_step_vertical_line_at_end(axs[1])

    axs[0].set_xticks([1,2,3,4])
    axs[0].set_xlim([1, 4.1])
    axs[0].set_xticklabels([10,100,1000,10000], size=14)
    axs[0].tick_params(labelsize=12)
    axs[0].set_xlabel('Foveation duration [ms]', size=14)
    axs[0].set_ylabel('CDF', size=14)
    axs[1].set_xlabel('Saccade amplitude [dva]', size=14)
    axs[1].tick_params(labelsize=12)
    axs[1].set_xlim([0, 20])
    sns.despine()
    handles, labels = axs[1].get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor(), lw=h.get_linewidth()) for h in handles]
    plt.legend(handles=new_handles, labels=labels, loc="lower right", fontsize=12)
    plt.show()


def check_grid_result_completeness(grid_identifier, nseeds=5, nvideos=10, path="/scratch/vito/scanpathes/results/", exclude_in_name=[]):
    nresults = nseeds * nvideos
    RES_DIRECTORIES = sorted(glob.iglob(f"{path}*"))[::-1]
    complete_dirs = []
    grid_dirs = [x for x in RES_DIRECTORIES if grid_identifier in x]
    if len(exclude_in_name) > 0:
        grid_dirs = [x for x in grid_dirs if not any([excl in x for excl in exclude_in_name])]
    for RES_DIR in grid_dirs:
        res_files = []
        for path, subdirs, files in os.walk(RES_DIR):
            for name in files:
                if name == "raw_results.pickle4":
                    res_files.append(os.path.join(path, name))
        if len(res_files) != nresults: 
            print(f"WARNING! Only {len(res_files)}/{nresults} result files in {RES_DIR}")
        else:
            complete_dirs.append(RES_DIR)
    # print("All good :)")
    return complete_dirs


def plot_emerging_stats(df_eval, ground_truth=None, ang_bins=30, sma_ws=5,
                        colors=["xkcd:blue", "xkcd:red"], summary_measure="mean", 
                        custom_name="Base model", obj_maxframes=90):
    
    fig, axs = plt.subplots(1, 2, dpi=150, figsize=(8, 3), sharey=False)
    kwargs = [{"alpha": 1, "lw": 3, "label": custom_name}]
    bdir_kwargs = [{"alpha": 1, "lw": 3}]

    if ground_truth is None:
        dfs = [df_eval]
    else:
        df_gt = ground_truth[ground_truth["video"].isin(df_eval["video"].unique())]
        dfs = [df_eval, df_gt]
        kwargs.append({"alpha": 1, "lw": 3, "label": "Humans"})
        bdir_kwargs.append({"alpha": 1, "lw": 2, "ls": "--"})

    for i, df in enumerate(dfs):
        x_vals, all = fovdur_vs_sacang(df, ang_bins, sma_ws, summary_measure)
        axs[1].plot(x_vals, all, color=colors[i], **kwargs[i])

    axs[1].set_xlabel('Change in saccade direction [°]', size=13)
    axs[1].set_xlim(-180, 180)
    # axs[0].set_yticks([200, 300, 400, 500, 600])
    axs[1].set_xticks([-180, -90, 0, 90, 180])
    axs[1].set_ylabel(f'Fov. dur. ({summary_measure}) [ms]', size=13)
    axs[1].tick_params(labelsize=12)
    axs[1].legend(labelspacing=0.3, prop={'size':9}, frameon=False)

    cols = ["xkcd:maroon", sns.color_palette("Dark2")[1], sns.color_palette("Dark2")[5], sns.color_palette("Dark2")[6]]
    labels = ["Background", "Detection", "Inspection", "Return"]
    for i, df in enumerate(dfs):
        ratios = get_BDIR_per_frames(df, obj_maxframes)
        for j in range(4):
            axs[0].plot(ratios[j], color=cols[j], **bdir_kwargs[i])
    axs[0].set_ylabel("Percentage", size=13)
    axs[0].set_xlabel("Time [frames]", size=13)
    axs[0].tick_params(labelsize=12)
    legend_elements = [Line2D([0], [0], color=cols[i], lw=2, label=labels[i]) for i in range(4)]
    # make legend smaller
    axs[0].legend(handles=legend_elements, loc="upper right", labelspacing=0.3, prop={'size':9}, frameon=False)  #, bbox_to_anchor=(1.2, 1.0))
    sns.despine(fig)
    fig.tight_layout()
    plt.show()


def eval_runs_to_dataframe_4d_grid_all(runs, df_gt, par1str="task", par2str="entropy", par3str="dv", par4str="sig"):
    data = {
        "mean_fd": {},
        "median_fd": {},
        "mean_sa": {},
        "median_sa": {},
        "fwd_sac": {},
        "bwd_sac": {},
        "ks_amp": {},
        "ks_dur": {},
        "obj_slope": {},
        "obj_r2": {},
        "fovcat_dist": {},
        "fit_fd_sa": {},
        "fit_fd_sa_fc": {},
        "fit_fd_sa_r2": {},
    }
    df_gt_obj = evaluate_all_obj(df_gt, prefix="gt_")
    gt_amp_dva = df_gt["sac_amp_dva"].dropna().values
    gt_dur_ms = df_gt["duration_ms"].dropna().values
    gt_fovcats = np.mean(get_BDIR_per_frames(df_gt, 90), axis=1)/100

    for run in runs:
        file_path = f"{run}/df_res_fov.csv"
        par1 = float(run.split("_")[-4][len(par1str):])
        par2 = float(run.split("_")[-3][len(par2str):])
        par3 = float(run.split("_")[-2][len(par3str):])
        par4 = float(run.split("_")[-1][len(par4str):])

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            data["mean_fd"][(par1, par2, par3, par4)] = np.nanmean(df["duration_ms"])
            data["median_fd"][(par1, par2, par3, par4)] = np.nanmedian(df["duration_ms"])
            data["mean_sa"][(par1, par2, par3, par4)] = np.nanmean(df["sac_amp_dva"])
            data["median_sa"][(par1, par2, par3, par4)] = np.nanmedian(df["sac_amp_dva"])
            prev_ang = df["sac_ang_p"].dropna().values
            data["fwd_sac"][(par1, par2, par3, par4)] = np.sum(np.abs(prev_ang)<10) / len(prev_ang)
            data["bwd_sac"][(par1, par2, par3, par4)] = np.sum(np.abs(prev_ang)>170) / len(prev_ang)
            # Fitness calculations
            if len(df["sac_amp_dva"].dropna().values) == 0:
                print(f"WARNING: No saccade amplitudes in {run}")
                ks_amp, p_amp = np.nan, np.nan
            else:
                ks_amp, p_amp = stats.ks_2samp(gt_amp_dva, df["sac_amp_dva"].dropna().values)
                ks_dur, p_dur = stats.ks_2samp(gt_dur_ms, df["duration_ms"].dropna().values)
            data["ks_amp"][(par1, par2, par3, par4)] = ks_amp
            data["ks_dur"][(par1, par2, par3, par4)] = ks_dur
            data["fit_fd_sa"][(par1, par2, par3, par4)] = (ks_amp + ks_dur) / 2 
            # Object fit
            df_obj = evaluate_all_obj(df, prefix="sim_")
            merged_df = pd.merge(df_gt_obj, df_obj, how="outer", on=['video', 'gt_object'])
            merged_df = merged_df[merged_df.gt_object != "nan"]
            merged_df = merged_df.fillna(0)
            slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['gt_tot_t'], merged_df['sim_tot_t'])
            data["obj_slope"][(par1, par2, par3, par4)] = slope
            data["obj_r2"][(par1, par2, par3, par4)] = r_value**2
            fovcat_dist = np.sum(np.abs(np.mean(get_BDIR_per_frames(df, 90), axis=1)/100 - gt_fovcats))
            data["fovcat_dist"][(par1, par2, par3, par4)] = fovcat_dist
            data["fit_fd_sa_fc"][(par1, par2, par3, par4)] = (ks_amp + ks_dur + fovcat_dist) / 3 
            data["fit_fd_sa_r2"][(par1, par2, par3, par4)] = (ks_amp + ks_dur + (1-r_value**2)) / 3 

        else:
            for key in data.keys():
                data[key][(par1, par2, par3, par4)] = np.nan    

    df_res = pd.DataFrame(
        {
            f"({par1str}, {par2str}, {par3str}, {par4str})": list(
                data["mean_fd"].keys()
            ),
            "mean_fd": list(data["mean_fd"].values()),
            "median_fd": list(data["median_fd"].values()),
            "mean_sa": list(data["mean_sa"].values()),
            "median_sa": list(data["median_sa"].values()),
            "fwd_sac": list(data["fwd_sac"].values()),
            "bwd_sac": list(data["bwd_sac"].values()),
            "ks_amp": list(data["ks_amp"].values()),
            "ks_dur": list(data["ks_dur"].values()),
            "obj_slope": list(data["obj_slope"].values()),
            "obj_r2": list(data["obj_r2"].values()),
            "fovcat_dist": list(data["fovcat_dist"].values()),
            "fit_fd_sa": list(data["fit_fd_sa"].values()),
            "fit_fd_sa_fc": list(data["fit_fd_sa_fc"].values()),
            "fit_fd_sa_r2": list(data["fit_fd_sa_r2"].values()),
        }
    )
    return df_res.sort_values(
        by=f"({par1str}, {par2str}, {par3str}, {par4str})", ascending=True
    ).reset_index(drop=True)


def plot_4d_grid(df_res, df_gt, metric="fov_dur", save_name=None):
    parcol = df_res.columns[0]
    p1 = parcol.split(",")[0][1:]
    p2 = parcol.split(",")[1][1:]
    p3 = parcol.split(",")[2][1:]
    p4 = parcol.split(",")[3][1:-1]

    p1_values, p2_values, p3_values, p4_values = zip(*df_res[parcol])
    # add arbitrary threshold here?!
    # p1_values = [x for x in p1_values if x < 0.9]
    p1_unival = np.array(np.unique(np.array(p1_values)))
    p2_unival = np.array(np.unique(np.array(p2_values)))
    p3_unival = np.array(np.unique(np.array(p3_values)))
    p4_unival = np.array(np.unique(np.array(p4_values)))

    fig, axs = plt.subplots(len(p1_unival), len(p2_unival), figsize=(3+2*len(p1_unival), 2+2*len(p2_unival)), sharex=True, sharey=True)
    cmap_q1, cmap_q2 = 'PiYG_r', 'RdBu_r'

    if metric == "fov_dur":
        fig.suptitle('Foveation Duration', fontsize=16)
        mean_gt = np.nanmean(df_gt["duration_ms"])
        median_gt = np.nanmedian(df_gt["duration_ms"])
        c_label_q1 = f'Median fov.dur. (GT: {round(median_gt, 2)}) [ms]'
        c_label_q2 = f'Mean fov.dur. (GT: {round(mean_gt, 2)}) [ms]'
        vmin_q1, vmax_q1 = -100, 100
        vmin_q2, vmax_q2 = -200, 200
    elif metric == "sac_amp":
        fig.suptitle('Saccade Amplitude', fontsize=16)
        mean_gt = np.nanmean(df_gt["sac_amp_dva"])
        median_gt = np.nanmedian(df_gt["sac_amp_dva"])
        c_label_q1 = f'Median sac.amp. (GT: {round(median_gt, 2)}) [dva]'
        c_label_q2 = f'Mean sac.amp. (GT: {round(mean_gt, 2)}) [dva]'
        vmin_q1, vmax_q1 = -1, 1
        vmin_q2, vmax_q2 = -2, 2
    elif metric == "prev_ang":
        fig.suptitle('Relative Saccade Angle', fontsize=16)
        prev_ang_gt = df_gt["sac_ang_p"].dropna().values
        prev_ang_gt_fwd = np.sum(np.abs(prev_ang_gt)<10) / len(prev_ang_gt)
        prev_ang_gt_bwd = np.sum(np.abs(prev_ang_gt)>170) / len(prev_ang_gt)
        c_label_q1 = f'Bwd. sac. (170-190°) (GT: {round(prev_ang_gt_bwd*100, 1)}) [%]'
        c_label_q2 = f'Fwd. sac. (-10-10°) (GT: {round(prev_ang_gt_fwd*100, 1)}) [%]'
        vmin_q1, vmax_q1 = -7, 7
        vmin_q2, vmax_q2 = -7, 7
    elif metric == "obj_fit":
        fig.suptitle('Fit to Object Dwell Time', fontsize=16)
        c_label_q1 = 'R2 value of fit'
        c_label_q2 = 'Slope of fit'
        vmin_q1, vmax_q1 = 0.5, 1
        vmin_q2, vmax_q2 = 0.5, 1.5
        cmap_q1 = 'pink'
    elif metric == "fitness":
        fig.suptitle('Fitness (KS fov.dur. & sac.amp.)', fontsize=16)
        c_label_q1 = 'KS fov. dur.'
        c_label_q2 = 'KS sac. amp.'
        vmin_q1, vmax_q1 = 0.0, 0.3
        vmin_q2, vmax_q2 = 0.0, 0.3
        cmap_q1, cmap_q2 = 'Blues', 'Reds'
    else:
        raise ValueError("Invalid metric specified.")

    for row, p1_val in enumerate(p1_unival):
        for col, p2_val in enumerate(p2_unival):
            df_p12 = df_res[(df_res[parcol].str[0] == p1_val) & (df_res[parcol].str[1] == p2_val)]
            if not df_p12.empty:
                _, _, p3_12_values, p4_12_values = zip(*df_p12[parcol])
            else:
                p3_12_values = [np.nan]
                p4_12_values = [np.nan]
            p3_12_values = np.array(p3_12_values)
            p4_12_values = np.array(p4_12_values)
            
            if metric == "fov_dur":
                c_q1_values = df_p12["median_fd"] - median_gt
                c_q2_values = df_p12["mean_fd"] - mean_gt

            elif metric == "sac_amp":
                c_q1_values = df_p12["median_sa"] - median_gt
                c_q2_values = df_p12["mean_sa"] - mean_gt

            elif metric == "prev_ang":
                c_q1_values = (df_p12["bwd_sac"] - prev_ang_gt_bwd) * 100
                c_q2_values = (df_p12["fwd_sac"] - prev_ang_gt_fwd) * 100

            elif metric == "obj_fit":
                c_q1_values = df_p12["obj_r2"]
                c_q2_values = df_p12["obj_slope"]

            elif metric == "fitness":
                c_q1_values = df_p12["ks_dur"]
                c_q2_values = df_p12["ks_amp"]

            scatter_q1 = axs[row, col].scatter(p3_12_values, p4_12_values + (p4_12_values.max() - p4_12_values.min()) / 40, c=c_q1_values, cmap=cmap_q1, marker='o', vmin=vmin_q1, vmax=vmax_q1, alpha=0.9)
            axs[row, col].set_xticks(p3_unival)
            axs[row, col].set_yticks(p4_unival)
            axs[row, col].set_title(f'{p1}={p1_val}, {p2}={p2_val}', fontsize=10)
            scatter_q2 = axs[row, col].scatter(p3_12_values, p4_12_values - (p4_12_values.max() - p4_12_values.min()) / 40, c=c_q2_values, cmap=cmap_q2, marker='o', vmin=vmin_q2, vmax=vmax_q2, alpha=0.9)
    
    fig.supxlabel(label_dict[p3])
    fig.supylabel(label_dict[p4])

    plt.tight_layout()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.3])
    fig.colorbar(scatter_q2, cax=cbar_ax, label=c_label_q2)
    cbar_ax = fig.add_axes([0.85, 0.55, 0.01, 0.3])
    fig.colorbar(scatter_q1, cax=cbar_ax, label=c_label_q1)
    plt.show()
    
    if save_name is not None:
        fig.savefig(f'{metric}_4d_grid_{save_name}.png', dpi=fig.dpi)

def plot_all_4d_grid(df_res, df_gt):
    for metric in ["fov_dur", "sac_amp", "prev_ang", "obj_fit", "fitness"]:
        plot_4d_grid(df_res, df_gt, metric)


def eval_runs_to_dataframe_grid_all(runs, df_gt, par1str="dv", par2str="sig"):
    data = {"mean_fd": {}, "median_fd": {}, "mean_sa": {}, "median_sa": {}, "fwd_sac": {}, "bwd_sac": {}, "obj_slope": {}, "obj_r2": {}}
    df_gt_obj = evaluate_all_obj(df_gt, prefix="gt_")

    for run in runs:
        # file_path = f"/scratch/vito/scanpathes/results/{run}/df_res_fov.csv"
        file_path = os.path.join(run, "df_res_fov.csv")
        par1 = float(run.split("_")[-2][len(par1str):])
        par2 = float(run.split("_")[-1][len(par2str):])
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            data["mean_fd"][(par1, par2)] = np.nanmean(df["duration_ms"])
            data["median_fd"][(par1, par2)] = np.nanmedian(df["duration_ms"])
            data["mean_sa"][(par1, par2)] = np.nanmean(df["sac_amp_dva"])
            data["median_sa"][(par1, par2)] = np.nanmedian(df["sac_amp_dva"])
            prev_ang = df["sac_ang_p"].dropna().values
            data["fwd_sac"][(par1, par2)] = np.sum(np.abs(prev_ang)<10) / len(prev_ang)
            data["bwd_sac"][(par1, par2)] = np.sum(np.abs(prev_ang)>170) / len(prev_ang)
            df_obj = evaluate_all_obj(df, prefix="sim_")
            merged_df = pd.merge(df_gt_obj, df_obj, how="outer", on=['video', 'gt_object'])
            merged_df = merged_df[merged_df.gt_object != "nan"]
            merged_df = merged_df.fillna(0)
            slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['gt_tot_t'], merged_df['sim_tot_t'])
            data["obj_slope"][(par1, par2)] = slope
            data["obj_r2"][(par1, par2)] = r_value**2
        else:
            for key in data.keys():
                data[key][(par1, par2)] = np.nan    
            
    df_res = pd.DataFrame({f"({par1str}, {par2str})": list(data["mean_fd"].keys()), 
                           "mean_fd": list(data["mean_fd"].values()), 
                           "median_fd": list(data["median_fd"].values()), 
                           "mean_sa": list(data["mean_sa"].values()), 
                           "median_sa": list(data["median_sa"].values()), 
                           "fwd_sac": list(data["fwd_sac"].values()), 
                           "bwd_sac": list(data["bwd_sac"].values()),
                           "obj_slope": list(data["obj_slope"].values()), 
                           "obj_r2": list(data["obj_r2"].values())})
    return df_res.sort_values(by=f"({par1str}, {par2str})", ascending=True).reset_index(drop=True)


# def plot_fov_dur_4d_grid(df_res, df_gt, save_name=None):
#     parcol = df_res.columns[0]
#     p1 = parcol.split(",")[0][1:]
#     p2 = parcol.split(",")[1][1:]
#     p3 = parcol.split(",")[2][1:]
#     p4 = parcol.split(",")[3][1:-1]
#     mean_gt = np.nanmean(df_gt["duration_ms"])
#     median_gt = np.nanmedian(df_gt["duration_ms"])

#     p1_values, p2_values, p3_values, p4_values = zip(*df_res[parcol])
#     p1_unival = np.array(np.unique(np.array(p1_values)))
#     p2_unival = np.array(np.unique(np.array(p2_values)))
#     p3_unival = np.array(np.unique(np.array(p3_values)))
#     p4_unival = np.array(np.unique(np.array(p4_values)))

#     fig, axs = plt.subplots(len(p1_unival), len(p2_unival), figsize=(8,7), sharex=True, sharey=True)

#     for row, p1_val in enumerate(p1_unival):
#         for col, p2_val in enumerate(p2_unival):
#             df_p12 = df_res[(df_res[parcol].str[0] == p1_val) & (df_res[parcol].str[1] == p2_val)]
#             _, _, p3_12_values, p4_12_values = zip(*df_p12[parcol])
#             p3_12_values = np.array(p3_12_values)
#             p4_12_values = np.array(p4_12_values)
#             # print(df_p12)

#             mean_fd_values = df_p12["mean_fd"] - mean_gt
#             median_fd_values = df_p12["median_fd"] - median_gt

#             scatter_median = axs[row,col].scatter(p3_12_values, p4_12_values + (p4_12_values.max()-p4_12_values.min())/40, c=median_fd_values, cmap='PiYG_r', marker='o', vmin=-100, vmax=100, alpha=0.9)
#             axs[row,col].set_xticks(p3_unival)
#             axs[row,col].set_yticks(p4_unival)
#             axs[row,col].set_title(f'{p1}={p1_val}, {p2}={p2_val}', fontsize=10)
#             scatter_mean = axs[row,col].scatter(p3_12_values, p4_12_values - (p4_12_values.max()-p4_12_values.min())/40, c=mean_fd_values, cmap='RdBu_r', marker='o', vmin=-200, vmax=200, alpha=0.9)
#     fig.supxlabel(label_dict[p3])
#     fig.supylabel(label_dict[p4])

#     # fig.colorbar(scatter_median, ax=axs.ravel().tolist(), label=f'Median fov.dur. (GT: {round(median_fd_gt, 2)})')
#     plt.tight_layout()
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.3])
#     fig.colorbar(scatter_mean, cax=cbar_ax, label=f'Mean fov.dur. (GT: {round(mean_gt, 2)}) [ms]')
#     cbar_ax = fig.add_axes([0.85, 0.55, 0.01, 0.3])
#     fig.colorbar(scatter_median, cax=cbar_ax, label=f'Median fov.dur. (GT: {round(median_gt, 2)}) [ms]')
#     plt.show()
#     if save_name is not None:
#         fig.savefig(f'fov_dur_4d_grid_{save_name}.png', dpi=fig.dpi)


# def plot_sac_amp_4d_grid(df_res, df_gt, save_name=None):
#     parcol = df_res.columns[0]
#     p1 = parcol.split(",")[0][1:]
#     p2 = parcol.split(",")[1][1:]
#     p3 = parcol.split(",")[2][1:]
#     p4 = parcol.split(",")[3][1:-1]
#     mean_gt = np.nanmean(df_gt["sac_amp_dva"])
#     median_gt = np.nanmedian(df_gt["sac_amp_dva"])

#     p1_values, p2_values, p3_values, p4_values = zip(*df_res[parcol])
#     p1_unival = np.array(np.unique(np.array(p1_values)))
#     p2_unival = np.array(np.unique(np.array(p2_values)))
#     p3_unival = np.array(np.unique(np.array(p3_values)))
#     p4_unival = np.array(np.unique(np.array(p4_values)))

#     fig, axs = plt.subplots(len(p1_unival), len(p2_unival), figsize=(8,7), sharex=True, sharey=True)

#     for row, p1_val in enumerate(p1_unival):
#         for col, p2_val in enumerate(p2_unival):
#             df_p12 = df_res[(df_res[parcol].str[0] == p1_val) & (df_res[parcol].str[1] == p2_val)]
#             _, _, p3_12_values, p4_12_values = zip(*df_p12[parcol])
#             p3_12_values = np.array(p3_12_values)
#             p4_12_values = np.array(p4_12_values)
#             # print(df_p12)

#             mean_values = df_p12["mean_sa"] - mean_gt
#             median_values = df_p12["median_sa"] - median_gt

#             scatter_median = axs[row,col].scatter(p3_12_values, p4_12_values + (p4_12_values.max()-p4_12_values.min())/40, c=median_values, cmap='PiYG_r', marker='o', vmin=-1, vmax=1, alpha=0.9)
#             axs[row,col].set_xticks(p3_unival)
#             axs[row,col].set_yticks(p4_unival)
#             axs[row,col].set_title(f'{p1}={p1_val}, {p2}={p2_val}', fontsize=10)
#             scatter_mean = axs[row,col].scatter(p3_12_values, p4_12_values - (p4_12_values.max()-p4_12_values.min())/40, c=mean_values, cmap='RdBu_r', marker='o', vmin=-1, vmax=1, alpha=0.9)
#     fig.supxlabel(label_dict[p3])
#     fig.supylabel(label_dict[p4])

#     plt.tight_layout()
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.3])
#     fig.colorbar(scatter_mean, cax=cbar_ax, label=f'Mean sac.amp. (GT: {round(mean_gt, 2)}) [dva]')
#     cbar_ax = fig.add_axes([0.85, 0.55, 0.01, 0.3])
#     fig.colorbar(scatter_median, cax=cbar_ax, label=f'Median sac.amp. (GT: {round(median_gt, 2)}) [dva]')
#     plt.show()
#     if save_name is not None:
#         fig.savefig(f'sac_amp_4d_grid_{save_name}.png', dpi=fig.dpi)


# def plot_prev_ang_4d_grid(df_res, df_gt, save_name=None):
#     parcol = df_res.columns[0]
#     p1 = parcol.split(",")[0][1:]
#     p2 = parcol.split(",")[1][1:]
#     p3 = parcol.split(",")[2][1:]
#     p4 = parcol.split(",")[3][1:-1]
#     prev_ang_gt = df_gt["sac_ang_p"].dropna().values
#     prev_ang_gt_fwd = np.sum(np.abs(prev_ang_gt)<10) / len(prev_ang_gt)
#     prev_ang_gt_bwd = np.sum(np.abs(prev_ang_gt)>170) / len(prev_ang_gt)


#     p1_values, p2_values, p3_values, p4_values = zip(*df_res[parcol])
#     p1_unival = np.array(np.unique(np.array(p1_values)))
#     p2_unival = np.array(np.unique(np.array(p2_values)))
#     p3_unival = np.array(np.unique(np.array(p3_values)))
#     p4_unival = np.array(np.unique(np.array(p4_values)))

#     fig, axs = plt.subplots(len(p1_unival), len(p2_unival), figsize=(8,7), sharex=True, sharey=True)

#     for row, p1_val in enumerate(p1_unival):
#         for col, p2_val in enumerate(p2_unival):
#             df_p12 = df_res[(df_res[parcol].str[0] == p1_val) & (df_res[parcol].str[1] == p2_val)]
#             _, _, p3_12_values, p4_12_values = zip(*df_p12[parcol])
#             p3_12_values = np.array(p3_12_values)
#             p4_12_values = np.array(p4_12_values)
#             # print(df_p12)

#             fwd_sac_values = (df_p12["fwd_sac"] - prev_ang_gt_fwd) * 100
#             bwd_sac_values = (df_p12["bwd_sac"] - prev_ang_gt_bwd) * 100

#             scatter_median = axs[row,col].scatter(p3_12_values, p4_12_values + (p4_12_values.max()-p4_12_values.min())/40, c=fwd_sac_values, cmap='PiYG_r', marker='o', vmin=-7, vmax=7, alpha=0.9)
#             axs[row,col].set_xticks(p3_unival)
#             axs[row,col].set_yticks(p4_unival)
#             axs[row,col].set_title(f'{p1}={p1_val}, {p2}={p2_val}', fontsize=10)
#             scatter_mean = axs[row,col].scatter(p3_12_values, p4_12_values - (p4_12_values.max()-p4_12_values.min())/40, c=bwd_sac_values, cmap='RdBu_r', marker='o', vmin=-7, vmax=7, alpha=0.9)
#     fig.supxlabel(label_dict[p3])
#     fig.supylabel(label_dict[p4])

#     # fig.colorbar(scatter_median, ax=axs.ravel().tolist(), label=f'Median fov.dur. (GT: {round(median_fd_gt, 2)})')
#     plt.tight_layout()
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.3])
#     fig.colorbar(scatter_mean, cax=cbar_ax, label=f'Fwd. sac. (-10-10°) (GT: {round(prev_ang_gt_fwd*100, 1)}) [%]')
#     cbar_ax = fig.add_axes([0.85, 0.55, 0.01, 0.3])
#     fig.colorbar(scatter_median, cax=cbar_ax, label=f'Bwd. sac. (170-190°) (GT: {round(prev_ang_gt_bwd*100, 1)}) [%]')
#     plt.show()
#     if save_name is not None:
#         fig.savefig(f'prev_ang_4d_grid_{save_name}.png', dpi=fig.dpi)


# def plot_obj_fit_4d_grid(df_res, df_gt, save_name=None):
#     parcol = df_res.columns[0]
#     p1 = parcol.split(",")[0][1:]
#     p2 = parcol.split(",")[1][1:]
#     p3 = parcol.split(",")[2][1:]
#     p4 = parcol.split(",")[3][1:-1]

#     p1_values, p2_values, p3_values, p4_values = zip(*df_res[parcol])
#     p1_unival = np.array(np.unique(np.array(p1_values)))
#     p2_unival = np.array(np.unique(np.array(p2_values)))
#     p3_unival = np.array(np.unique(np.array(p3_values)))
#     p4_unival = np.array(np.unique(np.array(p4_values)))

#     fig, axs = plt.subplots(len(p1_unival), len(p2_unival), figsize=(8,7), sharex=True, sharey=True)

#     for row, p1_val in enumerate(p1_unival):
#         for col, p2_val in enumerate(p2_unival):
#             df_p12 = df_res[(df_res[parcol].str[0] == p1_val) & (df_res[parcol].str[1] == p2_val)]
#             _, _, p3_12_values, p4_12_values = zip(*df_p12[parcol])
#             p3_12_values = np.array(p3_12_values)
#             p4_12_values = np.array(p4_12_values)
#             # print(df_p12)
#             scatter_median = axs[row,col].scatter(p3_12_values, p4_12_values + (p4_12_values.max()-p4_12_values.min())/40, c=df_p12["obj_r2"], cmap='pink', marker='o', vmin=0.5, vmax=1, alpha=0.9)
#             axs[row,col].set_xticks(p3_unival)
#             axs[row,col].set_yticks(p4_unival)
#             axs[row,col].set_title(f'{p1}={p1_val}, {p2}={p2_val}', fontsize=10)
#             scatter_mean = axs[row,col].scatter(p3_12_values, p4_12_values - (p4_12_values.max()-p4_12_values.min())/40, c=df_p12["obj_slope"], cmap='RdBu_r', marker='o', vmin=0.5, vmax=1.5, alpha=0.9)
#     fig.supxlabel(label_dict[p3])
#     fig.supylabel(label_dict[p4])

#     # fig.colorbar(scatter_median, ax=axs.ravel().tolist(), label=f'Median fov.dur. (GT: {round(median_fd_gt, 2)})')
#     plt.tight_layout()
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.3])
#     fig.colorbar(scatter_mean, cax=cbar_ax, label=f'Slope of fit')
#     cbar_ax = fig.add_axes([0.85, 0.55, 0.01, 0.3])
#     fig.colorbar(scatter_median, cax=cbar_ax, label=f'R2 value of fit')
#     plt.show()
#     if save_name is not None:
#         fig.savefig(f'obj_fit_4d_grid_{save_name}.png', dpi=fig.dpi)


# def plot_fitness_4d_grid(df_res, df_gt, save_name=None):
#     parcol = df_res.columns[0]
#     p1 = parcol.split(",")[0][1:]
#     p2 = parcol.split(",")[1][1:]
#     p3 = parcol.split(",")[2][1:]
#     p4 = parcol.split(",")[3][1:-1]

#     p1_values, p2_values, p3_values, p4_values = zip(*df_res[parcol])
#     # p2_values = [x for x in p2_values if x > 0.005]
#     p1_unival = np.array(np.unique(np.array(p1_values)))
#     p2_unival = np.array(np.unique(np.array(p2_values)))
#     p3_unival = np.array(np.unique(np.array(p3_values)))
#     p4_unival = np.array(np.unique(np.array(p4_values)))
#     # print(p1_unival, p2_unival, p3_unival, p4_unival)

#     fig, axs = plt.subplots(len(p1_unival), len(p2_unival), figsize=(8,7), sharex=True, sharey=True)

#     for row, p1_val in enumerate(p1_unival):
#         for col, p2_val in enumerate(p2_unival):
#             df_p12 = df_res[(df_res[parcol].str[0] == p1_val) & (df_res[parcol].str[1] == p2_val)]
#             # print(p1_val, p2_val, len(df_p12))
#             _, _, p3_12_values, p4_12_values = zip(*df_p12[parcol])
#             p3_12_values = np.array(p3_12_values)
#             p4_12_values = np.array(p4_12_values)
#             scatter_median = axs[row,col].scatter(p3_12_values, p4_12_values + (p4_12_values.max()-p4_12_values.min())/40, c=df_p12["ks_dur"], cmap='Blues', marker='o', vmin=0.0, vmax=0.3, alpha=0.9)
#             axs[row,col].set_xticks(p3_unival)
#             axs[row,col].set_yticks(p4_unival)
#             axs[row,col].set_title(f'{p1}={p1_val}, {p2}={p2_val}', fontsize=10)
#             scatter_mean = axs[row,col].scatter(p3_12_values, p4_12_values - (p4_12_values.max()-p4_12_values.min())/40, c=df_p12["ks_amp"], cmap='Reds', marker='o', vmin=0.0, vmax=0.3, alpha=0.9)
#     fig.supxlabel(label_dict[p3])
#     fig.supylabel(label_dict[p4])

#     # fig.colorbar(scatter_median, ax=axs.ravel().tolist(), label=f'Median fov.dur. (GT: {round(median_fd_gt, 2)})')
#     plt.tight_layout()
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.3])
#     fig.colorbar(scatter_mean, cax=cbar_ax, label=f'KS Amplitude')
#     cbar_ax = fig.add_axes([0.85, 0.55, 0.01, 0.3])
#     fig.colorbar(scatter_median, cax=cbar_ax, label=f'KS Duration')
#     plt.show()
#     if save_name is not None:
#         fig.savefig(f'fitness_4d_grid_{save_name}.png', dpi=fig.dpi)
