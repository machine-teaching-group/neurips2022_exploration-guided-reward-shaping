import numpy as np
import copy
import sys
import itertools as it
import os



import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib as mpl
mpl.rcParams['font.serif'] = ['times new roman']
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']

color_codes = ['r', 'g', 'b', '#F08080', "#8B0000", '#E802FB', "#C64C23",
 "#223206", "#7E3391", "#040004"]
# color_codes = [ 'r', 'g', 'b', '#F08080', "#8B0000", '#E802FB']
color_for_orig = "#8b0000"
color_for_pot = "#F08080"

mpl.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 34})
mpl.rc('legend', **{'fontsize': 30})
mpl.rc('text', usetex=True)
fig_size = [7, 4.8]

def plot_Q_learning(dict_file_q, figure_name, out_folder_name="out_folder", each_number=50):
    if not os.path.isdir(out_folder_name):
        os.makedirs(out_folder_name)

    plt.figure(3, figsize=fig_size)
    keys = [
            "expected_reward_env_orig_ExploRS",
            "expected_reward_env_orig_SelfRS",
            "expected_reward_env_orig_ExploB",
            "expected_reward_env_orig_SORS_with_Rbar",
            "expected_reward_env_orig_Orig",
    ]

    for key in keys:
        print("=========")
        print(key)

        if key == "expected_reward_env_orig_SORS_with_Rbar":
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{SORS}'$",
                         color='#FF8849', marker="<", ls=":", lw=4, markersize=10,
                         yerr=dict_file_q["SE_SORS_with_Rbar"][::each_number])


        elif key == "expected_reward_env_orig_SelfRS":
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{SelfRS}$",
                         color='#0a81ab', marker=">", ls="-.", markersize=10, lw=4,
                         yerr=dict_file_q["SE_SelfRS"][::each_number])


        elif key == "expected_reward_env_orig_ExploRS":
            # picked_states = dict_file[]
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{ExploRS}$",
                         color='g', marker=".", lw=4, markersize=18,
                         yerr=dict_file_q["SE_ExploRS"][::each_number])



        elif key == "expected_reward_env_orig_Orig":
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{Orig}$ ",
                         color="#8b0000", marker="^", ls="-.", markersize=10, lw=2.5,
                         yerr=dict_file_q["SE_Orig"][::each_number])

        elif key == "expected_reward_env_orig_ExploB":
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{ExploB}$ ",
                         color="cyan", marker="x", ls="-.", markersize=10, lw=2.5,
                         yerr=dict_file_q["SE_ExploB"][::each_number])


    # print(dict_file)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6), ncol=2, fancybox=False, shadow=False)
    plt.ylabel(r"Expected reward")
    plt.xlabel(r'Episode (x$10^{4}$)')
    # plt.xticks([0, 5, 10])
    # plt.yticks([0, 5, 10, 15])
    plt.yticks([0, 5, 10, 15])
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["0", "", "", "", "", "1.5", "", "", "", "", "3"])
    outFname = os.getcwd() + "/" + out_folder_name + "/{}.pdf".format(figure_name)
    print(outFname)
    plt.savefig(outFname, bbox_inches='tight')
    pass
#enddef

def input_from_file_Q(path, out_folder_name, n_file=10, t=100000):

    dict_file_q ={}

    std_matrix_Orig = []
    std_matrix_SelfRS = []
    std_matrix_ExploRS = []
    std_matrix_ExploB = []
    std_matrix_SORS_with_Rbar = []
    std_matrix_SORS_with_Rbar_Orig = []
    std_matrix_LIRPG_without_metagrad = []

    file_path = path
    expname = '/out_file_'
    for file in range(1, n_file + 1):
        with open(file_path + expname + str(file) + '.txt') as f:
            # with open(file_name) as f:
            print(file_path + expname + str(file) + '.txt')
            for line in f:
                read_line = line.split()
                if read_line[0] != '#':
                    if read_line[0] == 'initStates' or read_line[0] == 'active_state_opt_states' or read_line[
                        0] == 'sgd_state_opt_states' or read_line == 'template_list_for_initStates' or read_line[0] == 'first_succ_episode_number_teacher_Orig':
                        continue
                    elif read_line[0] in dict_file_q.keys():
                        dict_file_q[read_line[0]] += np.array(list(map(float, read_line[1:t])))
                    else:
                        dict_file_q[read_line[0]] = np.array(list(map(float, read_line[1:t])))
                    if read_line[0] == "expected_reward_env_orig_Orig":
                        std_matrix_Orig.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "expected_reward_env_orig_SelfRS":
                        std_matrix_SelfRS.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "expected_reward_env_orig_ExploRS":
                        std_matrix_ExploRS.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "expected_reward_env_orig_SORS_with_Rbar":
                        std_matrix_SORS_with_Rbar.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "expected_reward_env_orig_ExploB":
                        std_matrix_ExploB.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "expected_reward_env_orig_LIRPG_without_metagrad":
                        std_matrix_LIRPG_without_metagrad.append(np.array(list(map(float, read_line[1:t]))))


    std_Orig = np.std(std_matrix_Orig, axis=0)
    std_SelfRS = np.std(std_matrix_SelfRS, axis=0)
    std_ExploRS = np.std(std_matrix_ExploRS, axis=0)
    std_ExploB = np.std(std_matrix_ExploB, axis=0)
    std_SORS_with_Rbar = np.std(std_matrix_SORS_with_Rbar, axis=0)
    std_LIRPG_without_metagrad = np.std(std_matrix_LIRPG_without_metagrad, axis=0)

    SE_Orig = std_Orig / np.sqrt(len(std_matrix_Orig))
    SE_SelfRS= std_SelfRS / np.sqrt(len(std_matrix_SelfRS))
    SE_ExploRS = std_ExploRS / np.sqrt(len(std_matrix_ExploRS))
    SE_ExploB = std_ExploB / np.sqrt(len(std_matrix_ExploB))
    SE_SORS_with_Rbar = std_SORS_with_Rbar / np.sqrt(len(std_matrix_SORS_with_Rbar))
    SE_LIRPG_without_metagrad = std_LIRPG_without_metagrad / np.sqrt(len(std_matrix_LIRPG_without_metagrad))



    for key, value in dict_file_q.items():
        dict_file_q[key] = value / (n_file)

    dict_file_q["SE_Orig"] = np.array(SE_Orig)
    dict_file_q["SE_SelfRS"] = np.array(SE_SelfRS)
    dict_file_q["SE_ExploRS"] = np.array(SE_ExploRS)
    dict_file_q["SE_ExploB"] = np.array(SE_ExploB)
    dict_file_q["SE_SORS_with_Rbar"] = np.array(SE_SORS_with_Rbar)
    dict_file_q["SE_LIRPG_without_metagrad"] = np.array(SE_LIRPG_without_metagrad)
    return dict_file_q
#enndef

def plot_Q_learning_curves(path, out_folder_name, fig_name, n_file=10, t=100000):

    dict_to_plot = input_from_file_Q(path, out_folder_name, n_file=n_file, t=t)
    plot_Q_learning(dict_to_plot, fig_name, out_folder_name=out_folder_name, each_number=30)
    pass
#enddef



###############################
###############################
def plot_reinforce(dict_file_q, figure_name, out_folder_name="out_folder_name", each_number=30, four_room_flag=False):
    if not os.path.isdir(out_folder_name):
        os.makedirs(out_folder_name)


    plt.figure(3, figsize=fig_size)
    keys = [
            "expected_reward_env_orig_ExploRS",
            "expected_reward_env_orig_SelfRS",
            "expected_reward_env_orig_ExploB",
            "expected_reward_env_orig_SORS_with_Rbar",
            "expected_reward_env_orig_Orig",
             "expected_reward_env_orig_LIRPG_without_metagrad",
    ]

    for key in keys:
        print("=========")
        print(key)

        if key == "expected_reward_env_orig_LIRPG_without_metagrad":
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textnormal{LIRPG}'$",
                         color='#888686', marker="s", ls=":", lw=4, markersize=10,
                         yerr=dict_file_q["SE_LIRPG_without_metagrad"][::each_number])

        elif key == "expected_reward_env_orig_SORS_with_Rbar":
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{SORS}'$",
                         color='#FF8849', marker="<", ls=":", lw=4, markersize=10,
                         yerr=dict_file_q["SE_SORS_with_Rbar"][::each_number])


        elif key == "expected_reward_env_orig_SelfRS":
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{SelfRS}$",
                         color='#0a81ab', marker=">", ls="-.", markersize=10, lw=4,
                         yerr=dict_file_q["SE_SelfRS"][::each_number])


        elif key == "expected_reward_env_orig_ExploRS":
            # picked_states = dict_file[]
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{ExploRS}$",
                         color='g', marker=".", lw=4, markersize=18,
                         yerr=dict_file_q["SE_ExploRS"][::each_number])



        elif key == "expected_reward_env_orig_Orig":
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{Orig}$ ",
                         color="#8b0000", marker="^", ls="-.", markersize=10, lw=2.5,
                         yerr=dict_file_q["SE_Orig"][::each_number])

        elif key == "expected_reward_env_orig_ExploB":
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{ExploB}$ ",
                         color="cyan", marker="x", ls="-.", markersize=10, lw=2.5,
                         yerr=dict_file_q["SE_ExploB"][::each_number])


    # print(dict_file)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6), ncol=2, fancybox=False, shadow=False)
    plt.ylabel(r"Expected reward")
    plt.xlabel(r'Episode (x$10^{4}$)')
    plt.yticks([0, 5, 10, 15])
    if four_room_flag:
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["0", "", "1", "", "2", "", "3", "", "4", "", "5"])
    else:
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["0", "", "", "", "", "1.5", "", "", "", "", "3"])
    outFname = os.getcwd() + "/" + out_folder_name + "/{}.pdf".format(figure_name)
    print(outFname)
    plt.savefig(outFname, bbox_inches='tight')
    pass
#enddef

def input_from_file_reinforce(path, out_folder_name, n_file=10, t=100000):

    dict_file_q ={}

    std_matrix_orig = []
    std_matrix_SelfRS = []
    std_matrix_ExploRS = []
    std_matrix_ExploB = []
    std_matrix_SORS_with_Rbar = []
    std_matrix_SORS_with_Rbar_orig = []
    std_matrix_LIRPG_without_metagrad = []

    file_path = path
    expname = '/out_file_'
    for file in range(1, n_file + 1):
        with open(file_path + expname + str(file) + '.txt') as f:
            # with open(file_name) as f:
            print(file_path + expname + str(file) + '.txt')
            for line in f:
                read_line = line.split()
                if read_line[0] != '#':
                    if read_line[0] == 'initStates' or read_line[0] == 'active_state_opt_states' or read_line[
                        0] == 'sgd_state_opt_states' or read_line == 'template_list_for_initStates' or read_line[0] == 'first_succ_episode_number_teacher_orig':
                        continue
                    elif read_line[0] in dict_file_q.keys():
                        dict_file_q[read_line[0]] += np.array(list(map(float, read_line[1:t])))
                    else:
                        dict_file_q[read_line[0]] = np.array(list(map(float, read_line[1:t])))
                    if read_line[0] == "expected_reward_env_orig_Orig":
                        std_matrix_orig.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "expected_reward_env_orig_SelfRS":
                        std_matrix_SelfRS.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "expected_reward_env_orig_ExploRS":
                        std_matrix_ExploRS.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "expected_reward_env_orig_SORS_with_Rbar":
                        std_matrix_SORS_with_Rbar.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "expected_reward_env_orig_ExploB":
                        std_matrix_ExploB.append(np.array(list(map(float, read_line[1:t]))))
                    if read_line[0] == "expected_reward_env_orig_LIRPG_without_metagrad":
                        std_matrix_LIRPG_without_metagrad.append(np.array(list(map(float, read_line[1:t]))))


    std_orig = np.std(std_matrix_orig, axis=0)
    std_SelfRS = np.std(std_matrix_SelfRS, axis=0)
    std_ExploRS = np.std(std_matrix_ExploRS, axis=0)
    std_ExploB = np.std(std_matrix_ExploB, axis=0)
    std_SORS_with_Rbar = np.std(std_matrix_SORS_with_Rbar, axis=0)
    std_LIRPG_without_metagrad = np.std(std_matrix_LIRPG_without_metagrad, axis=0)

    SE_orig = std_orig / np.sqrt(len(std_matrix_orig))
    SE_SelfRS= std_SelfRS / np.sqrt(len(std_matrix_SelfRS))
    SE_ExploRS = std_ExploRS / np.sqrt(len(std_matrix_ExploRS))
    SE_ExploB = std_ExploB / np.sqrt(len(std_matrix_ExploB))
    SE_SORS_with_Rbar = std_SORS_with_Rbar / np.sqrt(len(std_matrix_SORS_with_Rbar))
    SE_LIRPG_without_metagrad = std_LIRPG_without_metagrad / np.sqrt(len(std_matrix_LIRPG_without_metagrad))



    for key, value in dict_file_q.items():
        dict_file_q[key] = value / (n_file)

    dict_file_q["SE_Orig"] = np.array(SE_orig)
    dict_file_q["SE_SelfRS"] = np.array(SE_SelfRS)
    dict_file_q["SE_ExploRS"] = np.array(SE_ExploRS)
    dict_file_q["SE_ExploB"] = np.array(SE_ExploB)
    dict_file_q["SE_SORS_with_Rbar"] = np.array(SE_SORS_with_Rbar)
    dict_file_q["SE_LIRPG_without_metagrad"] = np.array(SE_LIRPG_without_metagrad)
    return dict_file_q
#enndef

def plot_reinforce_curves(path, out_folder_name, figure_name, n_file=10, t=100000, four_room_flag=False):

    dict_to_plot = input_from_file_reinforce(path, out_folder_name, n_file=n_file, t=t)
    plot_reinforce(dict_to_plot, figure_name, out_folder_name=out_folder_name, each_number=30, four_room_flag=four_room_flag)
    pass
#enddef




