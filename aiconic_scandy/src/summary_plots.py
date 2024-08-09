import glob
import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import cv2
from collections import Counter
from scipy import stats

import scanpath_utils as su
import data_processing_hq as dp


def vonmises_kde(data, kappa, n_bins=100):
    """Polar KDE for angular distribution using von Mises distribution.

    :param data: Angla values in radians
    :type data: numpy.ndarray
    :param kappa: Kappa parameter of the von Mises distribution
    :type kappa: int
    :param n_bins: Number of angle bins, defaults to 100
    :type n_bins: int, optional
    :return: Bins and KDE values
    :rtype: tuple
    """
    from scipy.special import i0
    bins = np.linspace(-np.pi, np.pi, n_bins)
    x = np.linspace(-np.pi, np.pi, n_bins)
    kde = np.exp(kappa * np.cos(x[:, None] - data[None, :])).sum(1) / (2 * np.pi * i0(kappa))
    kde /= np.trapz(kde, x=bins)
    return bins, kde


def plot_fov_dur_sac_amp_hists(df_eval, ground_truth=None, savedir=None, min_amp=0.5, max_amp=30,
                               colors=["xkcd:blue", "xkcd:red"], name_flag="", custom_name="Model"):
    """
    Function that plots the foveation duration and saccade amplitude distributions
    of the given dataframe.

    :param df: Dataframe with the results
    :type df: pd.DataFrame
    """
    if ground_truth is None:
        dfs = [df_eval]
    else:
        dfs = [df_eval, ground_truth]

    fig, axs = plt.subplots(1, 2, dpi=150, figsize=(9.5, 3), sharey=True)
    histtype = ["bars", "step"]
    filltype = [True, False]
    name = [custom_name, "Humans"]

    for i, df in enumerate(dfs):
        amp_dva = df["sac_amp_dva"].dropna().values
        if min_amp is not None:
            amp_dva = amp_dva[amp_dva > min_amp]
        dur_ms = df["duration_ms"].dropna().values
        label = f"{name[i]}: mean={round(np.mean(dur_ms), 1)}, median={round(np.median(dur_ms), 1)}"
        fd_bins = np.linspace(1, 4, 50)
        sns.histplot(data=np.log10(dur_ms), kde=False, ax=axs[0], bins=fd_bins, color=colors[i], element=histtype[i],
                     fill=filltype[i], lw=2, label=label)
        # axs[0].set_title(f'FOV Duration: mean={round(np.mean(dur_ms),1)}, median={round(np.median(dur_ms),1)}')
        sa_bins = np.linspace(0, max_amp, int(max_amp*2))
        label = f"{name[i]}: mean={round(np.mean(amp_dva), 3)}, median={round(np.median(amp_dva), 3)}"
        sns.histplot(data=amp_dva, kde=False, ax=axs[1], bins=sa_bins, color=colors[i], element=histtype[i],
                     fill=filltype[i], lw=2,
                     label=label)
        # axs[1].set_title(f'SAC Amplitude: mean={round(np.mean(amp_dva), 3)}, median={round(np.median(amp_dva),3)}')

    axs[0].set_xticks([1, 2, 3, 4])
    axs[0].set_xticklabels([10, 100, 1000, 10000])
    axs[0].set_xlabel('Foveation duration [ms]', size=14)
    axs[0].set_ylabel('Count', size=14)
    axs[0].legend()
    axs[1].set_xlabel('Saccade amplitude [dva]', size=14)
    axs[1].set_xlim([0, max_amp])
    axs[0].tick_params(labelsize=12)
    axs[1].tick_params(labelsize=12)
    axs[1].legend()

    sns.despine(fig)
    fig.tight_layout()
    if savedir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(savedir, f'fov_dur_sac_amp_hists{name_flag}.png'))


def plot_sac_ang_hists(df_eval, ground_truth=None, savedir=None, ang_bins=30, colors=["xkcd:blue", "xkcd:red"], name_flag=""):
    fig, axs = plt.subplots(1, 2, dpi=150, figsize=(7, 3), subplot_kw={'projection': 'polar'})
    hori_ang = df_eval["sac_ang_h"].dropna().values
    prev_ang = df_eval["sac_ang_p"].dropna().values
    axs[0].hist(hori_ang / 180 * np.pi, ang_bins, density=True, color=colors[0])
    axs[1].hist(prev_ang / 180 * np.pi, ang_bins, density=True, color=colors[0])

    if ground_truth is not None:
        hori_ang = ground_truth["sac_ang_h"].dropna().values
        prev_ang = ground_truth["sac_ang_p"].dropna().values

        x_p, kde_p = vonmises_kde(hori_ang / 180 * np.pi, 50)
        axs[0].plot(x_p, kde_p, color=colors[1], lw=3)
        x_p, kde_p = vonmises_kde(prev_ang / 180 * np.pi, 50)
        axs[1].plot(x_p, kde_p, color=colors[1], lw=3)

    axs[0].set_xticks(axs[0].get_xticks())  # ; axs[0].set_yticks(axs[0].get_yticks())
    axs[0].set_xticklabels(["", "45°", "", "135°", "", "-135°", "", "-45°"])
    axs[0].set_title('Angle to horizontal')

    axs[1].set_xticks(axs[1].get_xticks())  # ; axs[1].set_yticks(axs[1].get_yticks())
    axs[1].set_xticklabels(["", "45°", "", "135°", "", "-135°", "", "-45°"])
    axs[1].set_title('Angle relative to previous')
    if savedir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(savedir, f'sac_ang_hists{name_flag}.png'))


def fovdur_vs_sacang(df_orig, num_bins, sma_ws, summary_measure='median'):
    """Prepares the data for the figure showing the relationship between saccade angle and binned foveation duration.

    :param df_orig: Original foveation dataframe (GT or SIM)
    :type df_orig: pandas.DataFrame
    :param num_bins: Number of bins to use for the saccade angle
    :type num_bins: int
    :param sma_ws: simple moving average window size
    :type sma_ws: int
    :return: List containing the x values (saccade angle bins) and the y values (mean foveation duration)
    :rtype: list
    """
    df = df_orig.copy()
    bins = np.linspace(-180, 180, num_bins + 1)
    x_vals = (bins[:-1] + bins[1:]) / 2

    ret = [x_vals]

    df['next_sac_ang_p'] = -1 * df['sac_ang_p'].shift(-1)
    df = df.dropna(subset=['next_sac_ang_p'])
    df['angle_bin'] = pd.cut(df['next_sac_ang_p'], bins=bins, labels=False)
    agg_df = df.groupby('angle_bin').agg({'duration_ms': [summary_measure, 'std']})
    agg_df.columns = ['mean_duration', 'std_duration']
    agg_df.reset_index(inplace=True)
    # agg_df = agg_df.append(agg_df.assign(angle_bin=agg_df['angle_bin'] + num_bins)).append(agg_df.assign(angle_bin=agg_df['angle_bin'] - num_bins))
    agg_df = pd.concat([agg_df, agg_df.assign(angle_bin=agg_df['angle_bin'] + num_bins),
                        agg_df.assign(angle_bin=agg_df['angle_bin'] - num_bins)], ignore_index=True)
    agg_df = agg_df.sort_values(by='angle_bin')
    agg_df[['mean_duration', 'std_duration']] = agg_df[['mean_duration', 'std_duration']].rolling(sma_ws,
                                                                                                  center=True).mean()
    agg_df = agg_df.iloc[num_bins:2 * num_bins]
    ret.append(agg_df['mean_duration'])
    return ret  # x_vals, agg_df['mean_duration']


def plot_ior_stats(df_eval, ground_truth=None, savedir=None, cutoff_t=3000, ret_bins=45, ang_bins=30, sma_ws=5,
                   colors=["xkcd:blue", "xkcd:red"], summary_measure="median", name_flag="", custom_name="Model"):
    fig, axs = plt.subplots(1, 2, dpi=150, figsize=(8, 3), sharey=False)
    kwargs = [{"alpha": 1, "lw": 2, "label": custom_name}]
    if ground_truth is None:
        dfs = [df_eval]
    else:
        dfs = [df_eval, ground_truth]
        kwargs.append({"alpha": 0.7, "lw": 3, "label": "Humans"})

    for i, df in enumerate(dfs):
        x_vals, all = fovdur_vs_sacang(df, ang_bins, sma_ws, summary_measure)
        axs[0].plot(x_vals, all, color=colors[i], **kwargs[i])

        bins = np.linspace(0, cutoff_t, ret_bins)
        counts, bins = np.histogram(df["ret_times"].dropna().values, bins)
        Nsac = len(df["sac_amp_dva"].dropna())
        axs[1].plot((bins[1:] + bins[:-1]) / 2, counts / Nsac, color=colors[i], **kwargs[i])
        # if i == 1:
        axs[1].legend()

    axs[0].set_xlabel('Change in saccade direction [°]', size=13)
    axs[0].set_xlim(-180, 180)
    # axs[0].set_yticks([200, 300, 400, 500, 600])
    axs[0].set_xticks([-180, -90, 0, 90, 180])
    axs[0].set_ylabel(f'Fov. dur. ({summary_measure}) [ms]', size=13)
    axs[0].tick_params(labelsize=12)
    axs[0].set_title('Space-based IOR metric')

    axs[1].set_xlabel('Return time [ms]', size=13)
    axs[1].set_ylabel('Percentage of saccades', size=13)  # Density
    # axs[1].set_yticks([0, 0.01, 0.02, 0.03])
    # axs[1].set_yticklabels([0, 1, 2, 3])
    axs[1].tick_params(labelsize=12)
    axs[1].set_title('Object-based IOR metric')

    sns.despine(fig)
    fig.tight_layout()
    if savedir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(savedir, f'ior_stats{name_flag}.png'))


def get_BDIR_per_frames(df_eval, maxframes=90):
    BDIR_per_frames = np.zeros((4, maxframes))
    bdir_to_row = {"B": 0, "D": 1, "I": 2, "R": 3, "-": 0}
    for index, row in df_eval.iterrows():
        if row["gt_object"] == "":
            continue
        BDIR_per_frames[bdir_to_row[row["fov_category"]], row["frame_start"]: max(maxframes - 1, row["frame_end"]) + 1, ] += 1
    BDIR_ratios_per_frames = 100 * BDIR_per_frames / np.sum(BDIR_per_frames, axis=0)
    # print(np.mean(BDIR_ratios_per_frames, axis=1))
    return BDIR_ratios_per_frames


def evaluate_all_obj(df_eval, prefix="", maxframes=0):
    """
    DIR evaluation based on individual objects
    """
    vid_sub_obj_list = []
    for videoname in df_eval.video.unique():
        df_vid = df_eval[df_eval["video"] == videoname]
        for sub in df_vid["subject"].unique():
            df_trial = df_vid[df_vid["subject"] == sub]
            if maxframes:
                df_trial = df_trial[df_trial["frame_start"] < maxframes]
            for obj_id in df_trial["gt_object"].unique():
                if obj_id in ["Ground", "", "nan"]:
                    continue
                # get all rows where this object was foveated
                dtemp = df_trial[df_trial["gt_object"] == obj_id]
                # add the overall time this object was foveated for each category
                d_cat = {}
                for fov_cat in ["D", "I", "R"]:
                    d_cat[fov_cat] = np.sum(dtemp["duration_ms"][dtemp["fov_category"] == fov_cat])
                vid_sub_obj_list.append(
                    {"video": videoname, "subject": sub, "gt_object": str(obj_id), "D": d_cat["D"], "I": d_cat["I"],
                        "R": d_cat["R"], })
    df_vso = pd.DataFrame(vid_sub_obj_list)
    # this gives us object statistics for each trial, based on this we now
    # calculate the average time spent and the ratio of subs for each category
    obj_list = []
    for video in df_vso.video.unique():
        df_vid = df_vso[df_vso["video"] == video]
        nsubj = len(df_vid.subject.unique())
        # go through all objects and calculate the average total time and ratios
        for obj in sorted(df_vid.gt_object.unique()):
            df_obj = df_vid[df_vid["gt_object"] == obj]
            d_r = len(df_obj) / nsubj
            i_r = len(df_obj[df_obj["I"] > 0]) / nsubj
            r_r = len(df_obj[df_obj["R"] > 0]) / nsubj
            d_t = df_obj["D"].sum() / nsubj
            i_t = df_obj["I"].sum() / nsubj
            r_t = df_obj["R"].sum() / nsubj
            tot_t = d_t + i_t + r_t
            obj_list.append(
                {"video": video, "gt_object": obj, f"{prefix}D_r": d_r, f"{prefix}I_r": i_r, f"{prefix}R_r": r_r,
                    f"{prefix}D_t": d_t, f"{prefix}I_t": i_t, f"{prefix}R_t": r_t, f"{prefix}tot_t": tot_t, })
    return pd.DataFrame(obj_list)


def plot_object_eval(df_eval, ground_truth=None, maxframes=90, savedir=None, cor_col="xkcd:green", name_flag=""):
    fig, axs = plt.subplots(1, 2, dpi=150, figsize=(8, 3), sharey=False)
    cols = ["xkcd:maroon", sns.color_palette("Dark2")[1], sns.color_palette("Dark2")[5], sns.color_palette("Dark2")[6]]
    labels = ["Background", "Detection", "Inspection", "Return"]
    kwargs = [{"alpha": 1, "lw": 2}]
    if ground_truth is None:
        dfs = [df_eval]
    else:
        dfs = [df_eval, ground_truth]
        kwargs.append({"alpha": 0.4, "lw": 3, "ls": "--"})
        df_gt_obj = evaluate_all_obj(ground_truth, prefix="gt_", maxframes=maxframes)
        df_obj = evaluate_all_obj(df_eval, prefix="sim_", maxframes=maxframes)
        merged_df = pd.merge(df_gt_obj, df_obj, how="outer", on=['video', 'gt_object'])
        merged_df = merged_df[merged_df.gt_object != "nan"]
        merged_df = merged_df.fillna(0)
        axs[1].scatter(merged_df['gt_tot_t'], merged_df['sim_tot_t'], marker='x', c=cor_col)
        axs[1].plot([0, merged_df['gt_tot_t'].max()], [0, merged_df['gt_tot_t'].max()], ls=":", color="k")
        slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['gt_tot_t'], merged_df['sim_tot_t'])
        print(
            f"slope: {slope}, intercept: {intercept}, r_value: {r_value}, p_value: {p_value}, std_err: {std_err}, r2 : {r_value ** 2}")
        axs[1].text(0.05, 0.95, r"$r^2$" + f"= {np.round(r_value ** 2, 2)}", transform=axs[1].transAxes, fontsize=10,
                    verticalalignment='top')
        axs[1].plot(merged_df['gt_tot_t'], intercept + slope * merged_df['gt_tot_t'], c=cor_col)

    for i, df in enumerate(dfs):
        ratios = get_BDIR_per_frames(df, maxframes)
        for j in range(4):
            axs[0].plot(ratios[j], color=cols[j], **kwargs[i])

    axs[0].set_ylabel("Percentage")
    axs[0].set_xlabel("Time [frames]")
    legend_elements = [Line2D([0], [0], color=cols[i], lw=2, label=labels[i]) for i in range(4)]
    axs[0].legend(handles=legend_elements)

    axs[1].set_ylabel("Model total dwell time [ms]")  # \n(mean for objects across runs)
    axs[1].set_xlabel("Human total dwell time [ms]")  # (mean for each object across subjects)
    axs[1].set_xticks([0, 1000, 2000, 3000])
    axs[1].set_yticks([0, 1000, 2000, 3000])
    # if ground_truth is not None:
    #     legend_df = plt.legend([Line2D([0], [0], color="k", **kwargs[i]) for i in range(2)], ["Model", "Human"], loc=4)
    #     plt.gca().add_artist(legend_df)

    sns.despine(fig)
    fig.tight_layout()
    if savedir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(savedir, f'obj_eval{name_flag}.png'))

def reduce_df_to_first_3sec(df):
    df = df[df['frame_start'] < 90]
    df.loc[df['frame_end'] > 90, 'duration_ms'] -= round(1000 / 30 * (df.loc[df['frame_end'] > 90, 'frame_end'] - 90))
    df.loc[df['frame_end'] > 90, 'sac_amp_dva'] = np.nan
    df.loc[df['frame_end'] > 90, 'sac_ang_h'] = np.nan
    df.loc[df['frame_end'] > 90, 'sac_ang_p'] = np.nan
    return df

def filter_dfs_for_eval_figures(df, df_gt, EVAL_ONLY_3SEC, dir_name):
    if EVAL_ONLY_3SEC:
        df = reduce_df_to_first_3sec(df)
        df_gt = reduce_df_to_first_3sec(df_gt)
    elif '_3SEC_' in dir_name:
        df_gt = reduce_df_to_first_3sec(df_gt)
    print(f'#Videos used: {len(df.video.unique())}')
    df_gt = df_gt[df_gt.video.isin(df.video.unique())]
    return df, df_gt


if __name__ == "__main__":
    EVAL_ONLY_3SEC = False
    RES_DIRECTORIES = sorted(glob.iglob("/media/vito/scanpath_backup/scanpath_results_current/*"))[::-1]
    # RES_DIRECTORIES = ["/media/vito/scanpath_backup/scanpath_results_current/2023-11-28-12-25-29_full"]  # single run
    for RES_DIR in RES_DIRECTORIES[:1]:
        print(RES_DIR)
        test_data = '_TEST_' in RES_DIR
        config_name = RES_DIR.split("/")[-1][19:]
        if EVAL_ONLY_3SEC:
            config_name = config_name + "_3secEval"
        res_files = []
        for path, subdirs, files in os.walk(RES_DIR):
            for name in files:
                if name == "raw_results.pickle4":
                    res_files.append(os.path.join(path, name))

        print("#Trials in simulation run: ", len(res_files))
        df = pd.concat([su.evaluate_model_trial(res) for res in res_files], ignore_index=True)
        df.to_csv(os.path.join(RES_DIR, 'df_res_fov.csv'))

        df_gt = pd.read_csv("/home/vito/Documents/eye_data_EM-UnEye_2023-11-29/df_res_gt_fov_all.csv")
        df, df_gt = filter_dfs_for_eval_figures(df, df_gt, EVAL_ONLY_3SEC, RES_DIR)

        # PLOTTING
        plot_fov_dur_sac_amp_hists(df, ground_truth=df_gt, savedir=RES_DIR, name_flag=config_name)
        plot_sac_ang_hists(df, ground_truth=df_gt, savedir=RES_DIR, name_flag=config_name)
        plot_ior_stats(df, ground_truth=df_gt, savedir=RES_DIR, name_flag=config_name)
        plot_object_eval(df, ground_truth=df_gt, savedir=RES_DIR, name_flag=config_name)

        # Run the frame-wise visualizations
        if EVAL_ONLY_3SEC is False:
            experiments = glob.glob(os.path.join(RES_DIR, "config_0", "*"))
            for p in experiments:
                dp.process_experiment_data(p, dpi=100)

