import dataclasses
import os
import subprocess
import socket

import torch
import glob
import datetime
import tqdm
import numpy as np

from run_experiment import ScanpathExperimentConfig, ScanpathExperiment
from experiment_util import dump_config


def run_over_sequences_for_general_config(
    base_config,
    sequence_bath_path,
    base_outputpath,
    video_directory=None,
    output_vis_vids=[],
    save_particles=False
):  # '5FU8vEKtyE'
    # Allow to only run for a single video
    if video_directory is None:
        pathes = glob.iglob(os.path.join(sequence_bath_path, "*"))
    else:
        pathes = [os.path.join(sequence_bath_path, video_directory)]

    for p in tqdm.tqdm(pathes):
        if not os.path.isdir(p):
            raise Warning("Path " + p + " is not a directoy!")
        sequence_name = p.split("/")[-1]
        output_path = os.path.join(base_outputpath, sequence_name)
        os.makedirs(output_path, exist_ok=False)
        config = dataclasses.replace(
            base_config, output_directory=output_path, video_directory=p
        )
        if sequence_name in output_vis_vids and config.seed == 1:
            config = dataclasses.replace(config, save_imgs=True, save_particles=save_particles)
        if not CLUSTER:
            with torch.no_grad():
                experiment = ScanpathExperiment()
            experiment.run(config)
            del experiment
            torch.cuda.empty_cache()
        else:
            slurm_path = os.path.join(output_path, "slurm")
            os.makedirs(slurm_path)
            config_file_path = os.path.join(slurm_path, "initial_config")
            config_file_path = dump_config(config, config_file_path)
            job_file_path = os.path.join(slurm_path, "job.bash")
            with open(job_file_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(
                    "#SBATCH --job-name=scanpath_experiment_"
                    + str(sequence_name)
                    + "\n"
                )
                f.write(
                    "#SBATCH --output=" + slurm_path + "/output.txt # output file\n"
                )
                f.write("#SBATCH --error=" + slurm_path + "/error.txt # output file\n")
                f.write("#SBATCH --time=0-05:00 # Runtime in D-HH:MM\n")
                f.write("#SBATCH --ntasks=1 # number of task remains 1\n")
                f.write(
                    "#SBATCH --partition=c0,c1a,c1b,c2 # Takes list, defaults to c2\n"
                ) 
                f.write("#SBATCH --cpus-per-task=4 # 1 task so we get 8 cpus\n")
                f.write("#SBATCH --mem-per-cpu=4000 # memory in MB per cpu allocated\n")
                f.write("source $HOME/miniconda3/bin/activate\n")
                f.write("conda activate cluster_config3\n")
                f.write(
                    "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/\n"
                )
                f.write(
                    "export PYTHONPATH=$PYTHONPATH:$HOME/miniconda3/envs/cluster_config3:$HOME/miniconda3/envs"
                    "/cluster_config3/lib/:/scratch/vito/scanpathes/code/:/scratch/vito/scanpathes/code"
                    "/segmentation_particle_filter_leightweight/src/:/scratch/vito/scanpathes/code"
                    "/domip_scanpathes/semantic_segmentation/FastSAM\n"
                )
                f.write("export YOLO_VERBOSE=False\n")
                f.write(
                    "python /scratch/vito/scanpathes/code/domip_scanpathes/src/run_experiment.py "
                    + config_file_path
                    + " ${TMPDIR}/local_cache\n"
                )
                f.write("exit\n")
            subprocess.Popen(["sbatch", job_file_path])


def create_start_configs_for_param_search(seeds=5, **kwargs):
    # base_config = ScanpathExperimentConfig()
    seed_configs = [
        ScanpathExperimentConfig(seed=s + 1, **kwargs) for s in range(seeds)
    ]
    # return [base_config]
    return seed_configs


def create_new_configs_for_param_search(old_configs, results):
    seed_configs = []
    decision_t = old_configs[0].decision_threshold + 1.0
    if decision_t > 4.1:
        return []
    for c in old_configs:
        seed_configs.append(
            ScanpathExperimentConfig(seed=c.seed, decision_threshold=decision_t)
        )
    return seed_configs


def create_start_configs_for_prompting_comparison():
    prompting_configs = [
        ScanpathExperimentConfig(seed=1),
        # ScanpathExperimentConfig(seed=2),
        # ScanpathExperimentConfig(seed=1, presaccadic_prompting=False),
        # ScanpathExperimentConfig(seed=2, presaccadic_prompting=False),
        # ScanpathExperimentConfig(seed=1, prompted_sam=False, presaccadic_prompting=False),
        # ScanpathExperimentConfig(seed=2, prompted_sam=False, presaccadic_prompting=False),
    ]
    return prompting_configs


def evaluate_results(configs):
    # TODO
    return []


config_obj_ablation_sets = {
    "full": create_start_configs_for_param_search(),
    "no_app": create_start_configs_for_param_search(use_app_seg=False),
    "no_motion": create_start_configs_for_param_search(use_motion_seg=False),
    "no_semantic": create_start_configs_for_param_search(use_semantic_seg=False),
    "no_prompts": create_start_configs_for_param_search(prompted_sam=False),
    "no_bottomup": create_start_configs_for_param_search(
        use_app_seg=False, use_motion_seg=False
    ),
    "no_topdown": create_start_configs_for_param_search(
        use_semantic_seg=False, prompted_sam=False
    ),
    # "no_sacmo_dv45": create_start_configs_for_param_search(saccadic_momentum=False, decision_threshold=4.5),
    # "no_sacmo_dv5": create_start_configs_for_param_search(saccadic_momentum=False, decision_threshold=5.0),
}
config_obj_ablation_nopresac_sets = {
    "full_no_presac": create_start_configs_for_param_search(
        presaccadic_prompting=False
    ),
    "no_app_no_presac": create_start_configs_for_param_search(
        use_app_seg=False, presaccadic_prompting=False
    ),
    "no_motion_no_presac": create_start_configs_for_param_search(
        use_motion_seg=False, presaccadic_prompting=False
    ),
    "no_semantic_no_presac": create_start_configs_for_param_search(
        use_semantic_seg=False, presaccadic_prompting=False
    ),
    # "no_prompts": create_start_configs_for_param_search(prompted_sam=False, presaccadic_prompting=False),
}
config_ior_ablation_sets = {
    "use_ior": create_start_configs_for_param_search(2, use_IOR_in_gaze_evidences=True),
    "no_uncert": create_start_configs_for_param_search(
        2, use_uncertainty_in_gaze_evidences=False
    ),
    "no_sacmo": create_start_configs_for_param_search(2, saccadic_momentum=False),
    "use_ior_no_sacmo": create_start_configs_for_param_search(
        2, use_IOR_in_gaze_evidences=True, saccadic_momentum=False
    ),
}
config_cb_sets = {
    "cb_no_sacmo_dv35": create_start_configs_for_param_search(
        center_bias=True, saccadic_momentum=False, decision_threshold=3.5
    ),
    "cb_no_sacmo_dv3": create_start_configs_for_param_search(
        center_bias=True, saccadic_momentum=False, decision_threshold=3.0
    ),
    "cb_no_sacmo_dv35_sig1": create_start_configs_for_param_search(
        center_bias=True,
        saccadic_momentum=False,
        decision_threshold=3.5,
        decision_noise=0.1,
    ),
    "cb_no_sacmo_dv3_sig1": create_start_configs_for_param_search(
        center_bias=True,
        saccadic_momentum=False,
        decision_threshold=3.0,
        decision_noise=0.1,
    ),
}
config_base_sets = {
    "cb_no_sacmo_dv3": create_start_configs_for_param_search(
        10,
        center_bias=True,
        saccadic_momentum=False,
        decision_threshold=3.0,
        decision_noise=0.3,
    ),
    "cb_no_sacmo_dv3_no_bottomup": create_start_configs_for_param_search(
        10,
        center_bias=True,
        saccadic_momentum=False,
        decision_threshold=3.0,
        decision_noise=0.3,
        use_app_seg=False,
        use_motion_seg=False,
    ),
    "cb_no_sacmo_dv3_no_topdown": create_start_configs_for_param_search(
        10,
        center_bias=True,
        saccadic_momentum=False,
        decision_threshold=3.0,
        decision_noise=0.3,
        use_semantic_seg=False,
        prompted_sam=False,
    ),
    "cb_no_sacmo_dv3_no_prompt": create_start_configs_for_param_search(
        10,
        center_bias=True,
        saccadic_momentum=False,
        decision_threshold=3.0,
        decision_noise=0.3,
        prompted_sam=False,
    ),
    "cb_no_sacmo_dv3_no_presac": create_start_configs_for_param_search(
        10,
        center_bias=True,
        saccadic_momentum=False,
        decision_threshold=3.0,
        decision_noise=0.3,
        presaccadic_prompting=False,
    ),
    "no_cb_sacmo_dv375_sig02": create_start_configs_for_param_search(
        10,
        center_bias=False,
        saccadic_momentum=True,
        decision_threshold=3.75,
        decision_noise=0.2,
    ),
}


# config_gridsearch_presac08_sets = {}
# for dv in np.linspace(2.6, 3.8, 7):
#     for sig in np.linspace(0.1, 0.4, 7):
#         dv = round(dv, 1)
#         sig = round(sig, 2)
#         key = f"cb_base_presac0.8_sens0.8_5seeds_dv{dv}_sig{sig}"
#         config_gridsearch_presac08_sets[key] = create_start_configs_for_param_search(5, center_bias=True, saccadic_momentum=False,
#                                                                             decision_threshold=dv, decision_noise=sig,
#                                                                             presaccadic_prompting=True, presaccadic_threshold=0.8*dv,
#                                                                             presaccadic_sensitivity=0.8,
#                                                                             use_uncertainty_in_gaze_evidences=True)

# config_gridsearch_presac1_sets = {}
# for dv in np.linspace(2.4, 3.6, 7):
#     for sig in np.linspace(0.1, 0.4, 7):
#         dv = round(dv, 1)
#         sig = round(sig, 2)
#         key = f"cb_base_presac0.8_sens1.0_5seeds_dv{dv}_sig{sig}"
#         config_gridsearch_presac1_sets[key] = create_start_configs_for_param_search(5, center_bias=True, saccadic_momentum=False,
#                                                                             decision_threshold=dv, decision_noise=sig,
#                                                                             presaccadic_prompting=True, presaccadic_threshold=0.8*dv,
#                                                                             presaccadic_sensitivity=1.0,
#                                                                             use_uncertainty_in_gaze_evidences=True)

# config_gridsearch_nopresac_amp_sets = {}
# for sens in np.array([1.0, 2.5, 10, 20.0]): #np.linspace(5.5, 7.5, 5):
#     for sig in np.linspace(0.15, 0.45, 7):
#         sens = round(sens, 1)
#         sig = round(sig, 2)
#         key = f"cb_base_nopresac_amp_dv3_5seeds_sens{sens}_sig{sig}"
#         config_gridsearch_nopresac_amp_sets[key] = create_start_configs_for_param_search(5, center_bias=True, saccadic_momentum=False,
#                                                                             decision_threshold=3.0, decision_noise=sig,
#                                                                             sensitivity_dva_sigma=sens,
#                                                                             presaccadic_prompting=False, #, presaccadic_threshold=0.8*dv, presaccadic_sensitivity=1.0,
#                                                                             use_uncertainty_in_gaze_evidences=True)

# range for uncert & sal: np.array([0.01, 0.05, 0.15, 0.3, 0.5, 0.75])
config_gridsearch_4d_sets = {}
for task in np.array([1.0]):  # 0.001, 0.05, 0.1, 0.2
    for entropy in np.array([0.25, 0.5, 1.0]):  # [0.001, 0.25, 0.5, 1.0] 0.001,
        for noise in np.array([0.2, 0.25, 0.3, 0.35, 0.4]):  # 0.2, 0.25, 0.3, , 0.4
            for threshold in np.array([2.0, 2.5, 3.0, 3.5, 4.0]):  # [2.0, 3.0, 4.0]
                # task = round(task, 2)
                # entropy = round(entropy, 2)
                key = f"cb_base_4d_grid_5seeds_task{task}_entropy{entropy}_dv{threshold}_sig{noise}"
                config_gridsearch_4d_sets[key] = create_start_configs_for_param_search(
                    5,
                    center_bias=True,
                    saccadic_momentum=False,
                    decision_threshold=threshold,
                    decision_noise=noise,
                    sensitivity_dva_sigma=6.0,
                    presaccadic_prompting=False,  # , presaccadic_threshold=0.8*dv, presaccadic_sensitivity=1.0,
                    use_uncertainty_in_gaze_evidences=True,
                    task_importance_added_number=task,
                    entropy_added_number=entropy,
                )


config_uncertainty_sets = {
    "task0.1_entropy0.5_dv3.0_sig0.3_noUncert": create_start_configs_for_param_search(
        5,
        center_bias=True,
        saccadic_momentum=False,
        use_uncertainty_in_gaze_evidences=False,
        decision_threshold=3.0,
        decision_noise=0.3,
        task_importance_added_number=0.1,
        entropy_added_number=0.5,
        presaccadic_prompting=False,
        # , presaccadic_sensitivity=0.8
    ),
    "task0.1_entropy0.6_dv3.0_sig0.3_noUncert": create_start_configs_for_param_search(
        5,
        center_bias=True,
        saccadic_momentum=False,
        use_uncertainty_in_gaze_evidences=False,
        decision_threshold=3.0,
        decision_noise=0.3,
        task_importance_added_number=0.1,
        entropy_added_number=0.6,
        presaccadic_prompting=False,
        # , presaccadic_sensitivity=0.8
    ),
    "task0.1_entropy0.7_dv3.0_sig0.3_noUncert": create_start_configs_for_param_search(
        5,
        center_bias=True,
        saccadic_momentum=False,
        use_uncertainty_in_gaze_evidences=False,
        decision_threshold=3.0,
        decision_noise=0.3,
        task_importance_added_number=0.1,
        entropy_added_number=0.7,
        presaccadic_prompting=False,
        # , presaccadic_sensitivity=0.8
    ),
}

config_best_sets = {
    "task0.001_entropy0.5_dv3.0_sig0.35_noUncert": create_start_configs_for_param_search(
        10,
        center_bias=True,
        saccadic_momentum=False,
        decision_threshold=3.0,
        decision_noise=0.35,
        task_importance_added_number=0.001,
        entropy_added_number=0.5,
        presaccadic_prompting=False,
        use_uncertainty_in_gaze_evidences=False,
    ),
    "task0.001_entropy0.5_dv3.0_sig0.35_momentumObj_1_2_15deg": create_start_configs_for_param_search(
        10,
        center_bias=True,
        saccadic_momentum=True,
        sac_momentum_on_obj=True,
        sac_momentum_min=1.0,
        sac_momentum_max=2.0,
        sac_momentum_restricted_angle=15,
        decision_threshold=3.0,
        decision_noise=0.35,
        task_importance_added_number=0.001,
        entropy_added_number=0.5,
        presaccadic_prompting=False,
    ),
    # "task0.001_entropy0.5_dv3.0_sig0.35": create_start_configs_for_param_search(10, center_bias=True, saccadic_momentum=False,
    #                                                                             decision_threshold=3.0,
    #                                                                             decision_noise=0.35,
    #                                                                             task_importance_added_number=0.001,
    #                                                                             entropy_added_number=0.5,
    #                                                                             presaccadic_prompting=False,
    #                                                                             # , presaccadic_sensitivity=0.8
    #                                                                             ),
    # "task0.05_entropy0.25_dv2.5_sig0.3": create_start_configs_for_param_search(10, center_bias=True,
    #                                                                             saccadic_momentum=False,
    #                                                                             decision_threshold=2.5,
    #                                                                             decision_noise=0.3,
    #                                                                             task_importance_added_number=0.05,
    #                                                                             entropy_added_number=0.25,
    #                                                                             presaccadic_prompting=False,
    #                                                                             # , presaccadic_sensitivity=0.8
    #                                                                             ),
    # "3SEC_GT_OBJECTS_task0.001_entropy0.5_dv3.0_sig0.35": create_start_configs_for_param_search(10, center_bias=True,
    #                                                                                        saccadic_momentum=False,
    #                                                                                        decision_threshold=3.0,
    #                                                                                        decision_noise=0.35,
    #                                                                                        task_importance_added_number=0.001,
    #                                                                                        entropy_added_number=0.5,
    #                                                                                        presaccadic_prompting=False,
    #                                                                                        use_ground_truth_objects=True,
    #                                                                                        stop_after_3sec=True,
    #                                                                                        ),
    # "3SEC_GT_OBJECTS_task0.05_entropy0.25_dv2.5_sig0.3": create_start_configs_for_param_search(10, center_bias=True,
    #                                                                                       saccadic_momentum=False,
    #                                                                                       decision_threshold=2.5,
    #                                                                                       decision_noise=0.3,
    #                                                                                       task_importance_added_number=0.05,
    #                                                                                       entropy_added_number=0.25,
    #                                                                                       presaccadic_prompting=False,
    #                                                                                       use_ground_truth_objects=True,
    #                                                                                       stop_after_3sec=True,
    #                                                                                       ),
    # "3SEC_GT_OBJECTS_NO_UNCERT_task0.001_entropy0.5_dv3.0_sig0.35": create_start_configs_for_param_search(10, center_bias=True,
    #                                                                                                  saccadic_momentum=False,
    #                                                                                                  decision_threshold=3.0,
    #                                                                                                  decision_noise=0.35,
    #                                                                                                  task_importance_added_number=0.001,
    #                                                                                                  entropy_added_number=0.5,
    #                                                                                                  presaccadic_prompting=False,
    #                                                                                                  use_ground_truth_objects=True,
    #                                                                                                  stop_after_3sec=True,
    #                                                                                                  use_uncertainty_in_gaze_evidences=False,
    #                                                                                                  ),
    # "3SEC_GT_OBJECTS_NO_UNCERT_task0.05_entropy0.25_dv2.5_sig0.3": create_start_configs_for_param_search(10, center_bias=True,
    #                                                                                                 saccadic_momentum=False,
    #                                                                                                 decision_threshold=2.5,
    #                                                                                                 decision_noise=0.3,
    #                                                                                                 task_importance_added_number=0.05,
    #                                                                                                 entropy_added_number=0.25,
    #                                                                                                 presaccadic_prompting=False,
    #                                                                                                 use_ground_truth_objects=True,
    #                                                                                                 stop_after_3sec=True,
    #                                                                                                 use_uncertainty_in_gaze_evidences=False,
    #                                                                                                 ),
}


    # config_new_noUncert = {
    #     "TEST_NO_UNCERT_task0.5_entropy1.0_dv5.0_sig0.1": create_start_configs_for_param_search(
    #         10,
    #         center_bias=True,
    #         saccadic_momentum=False,
    #         decision_threshold=5.0,
    #         decision_noise=0.1,
    #         task_importance_added_number=0.5,
    #         entropy_added_number=1.0,
    #         presaccadic_prompting=False,
    #         use_ground_truth_objects=False,
    #         stop_after_3sec=False,
    #         use_uncertainty_in_gaze_evidences=False,
    #     ),
    #     "TEST_NO_UNCERT_task0.5_entropy1.0_dv5.0_sig0.2": create_start_configs_for_param_search(
    #         10,
    #         center_bias=True,
    #         saccadic_momentum=False,
    #         decision_threshold=5.0,
    #         decision_noise=0.2,
    #         task_importance_added_number=0.5,
    #         entropy_added_number=1.0,
    #         presaccadic_prompting=False,
    #         use_ground_truth_objects=False,
    #         stop_after_3sec=False,
    #         use_uncertainty_in_gaze_evidences=False,
    #     ),
    #     "TEST_NO_UNCERT_task1.0_entropy1.0_dv6.0_sig0.1": create_start_configs_for_param_search(
    #         10,
    #         center_bias=True,
    #         saccadic_momentum=False,
    #         decision_threshold=6.0,
    #         decision_noise=0.1,
    #         task_importance_added_number=1.0,
    #         entropy_added_number=1.0,
    #         presaccadic_prompting=False,
    #         use_ground_truth_objects=False,
    #         stop_after_3sec=False,
    #         use_uncertainty_in_gaze_evidences=False,
    #     ),
    # }
    

def create_test_config_sets(
    task, entropy, dv, sig, base=False, noUncert=False, presac=False, sacmom=False, gtobj=False, gtobj_noUncert=False, hlobj=False, noglobal=False, llobj=False, noprompt=False, llprompt=False, llpsam=False
):
    config_TEST_sets = {}
    
    if base:
        key = f"TEST_base_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=False,
            use_ground_truth_objects=False,
            stop_after_3sec=False,
            use_uncertainty_in_gaze_evidences=True,
        )
    if noUncert:
        key = f"TEST_noUncert_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=False,
            use_ground_truth_objects=False,
            stop_after_3sec=False,
            use_uncertainty_in_gaze_evidences=False,
        )
    if presac:
        key = f"TEST_preSac0.6-1_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=True,
            presaccadic_sensitivity=1.0,
            presaccadic_threshold=dv * 0.6,
            use_ground_truth_objects=False,
            stop_after_3sec=False,
            use_uncertainty_in_gaze_evidences=True,
        )
        key = f"TEST_preSac0.3-1.0_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=True,
            presaccadic_sensitivity=1.0,
            presaccadic_threshold=dv*0.3,
            use_ground_truth_objects=False,
            stop_after_3sec=False,
            use_uncertainty_in_gaze_evidences=True,
        )
        key = f"TEST_preSac0.7-1.0_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=True,
            presaccadic_sensitivity=1.0,
            presaccadic_threshold=dv*0.7,
            use_ground_truth_objects=False,
            stop_after_3sec=False,
            use_uncertainty_in_gaze_evidences=True,
        )
    if sacmom:
        # key = f"TEST_sacMom0.9-2.0-20deg_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        key = f"TEST_sacMom0.85-2.5-35deg_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=True,
            sac_momentum_on_obj=True,
            sac_momentum_min=0.85,
            sac_momentum_max=2.5,
            sac_momentum_restricted_angle=35,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=False,
            use_ground_truth_objects=False,
            stop_after_3sec=False,
            use_uncertainty_in_gaze_evidences=True,
        )
    if gtobj:
        key = f"TEST_3SEC_gtObj_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=False,
            use_ground_truth_objects=True,
            stop_after_3sec=True,
            use_uncertainty_in_gaze_evidences=True,
        )
    if gtobj_noUncert:
        key = f"TEST_3SEC_gtObj_noUncert_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=False,
            use_ground_truth_objects=True,
            stop_after_3sec=True,
            use_uncertainty_in_gaze_evidences=False,
        )
    if hlobj:
        key = f"TEST_hlObj_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=False,
            use_motion_seg=False,
            use_app_seg=False,
            use_uncertainty_in_gaze_evidences=True,
        )
    if noglobal:
        key = f"TEST_noGlobalSAM_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=False,
            use_semantic_seg=False,
            prompted_sam=True,
            use_motion_seg=True,
            use_app_seg=True,
            use_uncertainty_in_gaze_evidences=True,
        )
    if llobj:
        key = f"TEST_llObj_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=False,
            use_semantic_seg=False,
            prompted_sam=False,
            use_uncertainty_in_gaze_evidences=True,
        )
    if noprompt:
        key = f"TEST_noPrompt_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=False,
            use_semantic_seg=True,
            prompted_sam=False,
            use_uncertainty_in_gaze_evidences=True,
        )
    if llprompt:
        key = f"TEST_llPrompt_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=False,
            use_semantic_seg=False,
            prompted_sam=True,
            use_uncertainty_in_gaze_evidences=True,
            use_low_level_prompt=True,
        )
    if llpsam:
        key = f"TEST_llPromptGlobalSAM_task{task}_entropy{entropy}_dv{dv}_sig{sig}"
        config_TEST_sets[key] = create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=False,
            use_semantic_seg=True,
            prompted_sam=True,
            use_uncertainty_in_gaze_evidences=True,
            use_low_level_prompt=True,
        )

    return config_TEST_sets

def create_base_test_config_sets(task, entropy, dv, sig):
    config_TEST_sets = {
        f"TEST_base_task{task}_entropy{entropy}_dv{dv}_sig{sig}": create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=False,
            use_ground_truth_objects=False,
            stop_after_3sec=False,
            use_uncertainty_in_gaze_evidences=True,
        ),
    }
    return config_TEST_sets

def create_noUncert_test_config_sets(task, entropy, dv, sig):
    config_TEST_sets = {
        f"TEST_noUncert_task{task}_entropy{entropy}_dv{dv}_sig{sig}": create_start_configs_for_param_search(
            10,
            center_bias=False,
            saccadic_momentum=False,
            decision_threshold=dv,
            decision_noise=sig,
            task_importance_added_number=task,
            entropy_added_number=entropy,
            sensitivity_dva_sigma=7.0,
            presaccadic_prompting=False,
            use_ground_truth_objects=False,
            stop_after_3sec=False,
            use_uncertainty_in_gaze_evidences=False,
        ),
    }
    return config_TEST_sets



if __name__ == "__main__":
    print("experiment runner started......")
    CLUSTER = socket.gethostname() != "scioi-1712"
    if CLUSTER:
        # base_path = "/scratch/vito/scanpathes/scanpath_data/training_data"
        base_path = "/scratch/vito/scanpathes/scanpath_data/test_data"
        results_path_general = "/scratch/vito/scanpathes/results"
        output_videos = []
    else:
        base_path = "/home/vito/scanpath_things/scanpath_data"
        # base_path = "/home/vito/scanpath_things/scanpath_data_test"
        results_path_general = "/media/vito/scanpath_backup/scanpath_results_current"
        output_videos = [] # ['0azWFsmoz38', '2DNG46ZD9Ss', '-UJgyiWe500']

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("Using CUDA")
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("NOT using CUDA")
        torch.set_default_tensor_type(torch.FloatTensor)

    config_gridsearch_4d_sets = {}
    # BIG grid
    # for task in np.array([0.0, 0.11, 0.25, 0.5]):  # 0.001, 0.05, 0.1, 0.2
    #     for entropy in np.array([0.11]):  # [0.001, 0.25, 0.5, 1.0] 0.001,
    #         for noise in np.array([0.1, 0.2, 0.3]):  # 0.2, 0.25, 0.3, , 0.4
    #             for threshold in np.array([3.0, 4.0, 5.0, 6.0]):  # [2.0, 2.5, 3.0, 3.5, 4.0]
    # FINE grid around best point
    all_runs = os.listdir("/scratch/vito/scanpathes/results/")
    for task in np.array([0.11, 0.25, 0.5]):  # 0.0, 0.11, 0.25, 0.5  # 0.0
        for entropy in np.array([0.0, 0.11, 0.25, 0.5, 1.0]):  # 0.0, 0.11, 0.25, 0.5
            for noise in np.array([0.4]): # 0.1, 0.2, 0.3  # 0.2, 0.25, 0.3, 0.35, 0.4 
                for threshold in np.array([2.0, 3.0, 4.0, 5.0, 6.0]):  # 2.0, 3.0, 4.0, 5.0, 6.0
                    key = f"base_4d_grid_5seeds_task{task}_entropy{entropy}_dv{threshold}_sig{noise}"
                    # key = f"final_base_4d_grid_5seeds_task{task}_entropy{entropy}_dv{threshold}_sig{noise}"
                    # key = f"gt_obj_grid_5seeds_task{task}_entropy{entropy}_dv{threshold}_sig{noise}" 
                    # check if a config with this key already exists in directory /scratch/vito/scanpathes/results/
                    if len([dir for dir in all_runs if dir.endswith(key)]) == 0:
                        config_gridsearch_4d_sets[key] = (
                            create_start_configs_for_param_search(
                                5,
                                center_bias=False,
                                saccadic_momentum=False,
                                decision_threshold=threshold,
                                decision_noise=noise,
                                sensitivity_dva_sigma=7.0,
                                presaccadic_prompting=False,
                                use_uncertainty_in_gaze_evidences=True,
                                # use_uncertainty_in_gaze_evidences=False,
                                task_importance_added_number=task,
                                entropy_added_number=entropy,
                                # # prompted_sam=False, # no_prompt_grid_5seeds_
                                # use_ground_truth_objects=True, # gt_obj_grid_5seeds
                                # stop_after_3sec=True, # gt_obj_grid_5seeds
                            )
                        )

    # config_sgl_visualize_set = {
    #     "BOUNCYCASTLE_task0.001_entropy0.5_dv3.0_sig0.0": create_start_configs_for_param_search(1, center_bias=True,
    #                                                                                            saccadic_momentum=False,
    #                                                                                            decision_threshold=3.0,
    #                                                                                            decision_noise=0.0,
    #                                                                                            task_importance_added_number=0.001,
    #                                                                                            entropy_added_number=0.5,
    #                                                                                            presaccadic_prompting=False,
    #                                                                                            use_ground_truth_objects=False,
    #                                                                                            stop_after_3sec=False,
    #                                                                                            # , presaccadic_sensitivity=0.8
    #                                                                                            ),
    # }

    # task, entropy, dv, sig = 0.5, 0.5, 4.0, 0.25  # 0.5, 1.0, 5.0, 0.2 # 0.2, 0.25, 3.0, 0.35
    # # TODO sets: original grid: 0.05, 0.25, 2.5, 0.3 -> already TEST; 0.1, 0.5, 3.5, 0.4 -> nothing yet
    # config_TEST_sets = create_test_config_sets(task, entropy, dv, sig, noUncert=True, gtobj=False, presac=False, sacmom=False)
    
    task, entropy, dv, sig = 0.0, 0.5, 4.0, 0.4 # 0.0, 0.25, 3.5, 0.4 #0.0, 0.0, 2.5, 0.25  #0.0, 0.5, 4.0, 0.4  
    # config_TEST_sets = create_base_test_config_sets(task, entropy, dv, sig)
    # config_TEST_sets = create_noUncert_test_config_sets(task, entropy, dv, sig, sacmom=True)
    # config_TEST_sets = create_test_config_sets(task, entropy, dv, sig, base=False, noUncert=False, presac=True, sacmom=True, gtobj=True, gtobj_noUncert=False, hlobj=True, noglobal=True, llobj=True, noprompt=True, llprompt=True, llpsam=True)
    config_TEST_sets = create_test_config_sets(task, entropy, dv, sig, presac=True) # , noprompt=True, llobj=True

    # print(config_gridsearch_4d_sets)
    for name, config_set in config_TEST_sets.items():
        if "TEST" in name:
            assert "test" in base_path, "TEST in name but not in base_path!"
        else:
            assert "test" not in base_path, "TEST not in name but in base_path!"
        output_path = os.path.join(
            results_path_general,
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_" + name,
        )
        # print(name, config_set, output_path)
        for i, c in enumerate(config_set):
            run_over_sequences_for_general_config(
                c,
                base_path,
                os.path.join(output_path, "config_" + str(i)),
                output_vis_vids=output_videos,
                save_particles=False,
            )  # ["-5FU8vEKtyE"]) #"07YX_GR_cEE", "24cgfaG8WI0", "-2HFZjPOCMk", "3JGPQ3loSV4", "-5FU8vEKtyE", "0AkA2Ru9qG0", "2fojVBo1tv0", "3-evHTgxa8M", "4-dI5vRgGWQ", "-sGcmYcU_QI"],
