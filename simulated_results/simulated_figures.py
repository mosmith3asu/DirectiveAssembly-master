import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
np.random.seed(0)

def reward_radar_plots():
    my_pal = {"coactive": "lightgray", "directive": "tab:blue"}
    # TRUST ###################################
    fig, axs = plt.subplots(1, 1,constrained_layout=True,)  # figsize=(10,5)


    coactive_survey = [1, 2, 3, 4, 5, 6, 7]
    directive_survey = [1, 2, 3, 4, 5, 6, 7]

    # df = pd.DataFrame({'item':survey_items,'coactive':coactive_survey,'directive':directive_survey})
    df = pd.DataFrame({'group': ['coactive','directive'],
                        '# Moves': [4.1, 8.1],
                        'Time Remaining': [6.6, 7.9],
                        'Reliable': [2.4, 7.5],
                        '(10-Unresponsive)': [2.8, 7.3],
                        'Act\nconsistently': [6.3, 5.8],
                        'Met the\nneeds of\nthe task': [6.1, 7.2],
                        'Perform as\nexpected': [5.7, 9.1]}
                      )



    # number of variable
    categories = list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6,8,10], ['2', '4', '6','8','10'], color="grey", size=7)
    plt.ylim(0, 10)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind2
    values = df.loc[1].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Directive",c=my_pal['directive'])
    ax.fill(angles, values, my_pal['directive'], alpha=0.6)# alpha=0.1)

    # Ind1
    values = df.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Coactive",c=my_pal['coactive'])
    ax.fill(angles, values, my_pal['coactive'], alpha=0.75)

    # Add legend
    # ax.set_rlabel_position(-102.5)  # Move radial labels away from plotted line
    ax.set_theta_offset(np.pi / 2)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # ax.set_thetagrids(values, frac=1.3)
    # ax.set_thetagrids(values, frac=1.3)
    ax.xaxis.set_tick_params(pad=18)
    # Show the graph
    plt.show()
def radar_plots():
    my_pal = {"coactive": "lightgray", "directive": "tab:blue"}
    # TRUST ###################################
    fig, axs = plt.subplots(1, 1,constrained_layout=True,)  # figsize=(10,5)


    coactive_survey = [1, 2, 3, 4, 5, 6, 7]
    directive_survey = [1, 2, 3, 4, 5, 6, 7]
    survey_items = ['Dependable', 'Reliable', '(10-Unresponsive)', 'Predictable','Act consistently',
                       'Met the needs of the task','Perform as expected']
    # df = pd.DataFrame({'item':survey_items,'coactive':coactive_survey,'directive':directive_survey})
    df = pd.DataFrame({'group': ['coactive','directive'],
                        'Dependable': [4.1, 8.1],
                       'Predictable': [6.6, 7.9],
                        'Reliable': [2.4, 7.5],
                        '(10-Unresponsive)': [2.8, 7.3],
                        'Act\nconsistently': [6.3, 5.8],
                        'Met the\nneeds of\nthe task': [6.1, 7.2],
                        'Perform as\nexpected': [5.7, 9.1]})



    # number of variable
    categories = list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6,8,10], ['2', '4', '6','8','10'], color="grey", size=7)
    plt.ylim(0, 10)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind2
    values = df.loc[1].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Directive",c=my_pal['directive'])
    ax.fill(angles, values, my_pal['directive'], alpha=0.6)# alpha=0.1)

    # Ind1
    values = df.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Coactive",c=my_pal['coactive'])
    ax.fill(angles, values, my_pal['coactive'], alpha=0.75)

    # Add legend
    # ax.set_rlabel_position(-102.5)  # Move radial labels away from plotted line
    ax.set_theta_offset(np.pi / 2)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # ax.set_thetagrids(values, frac=1.3)
    # ax.set_thetagrids(values, frac=1.3)
    ax.xaxis.set_tick_params(pad=18)
    # Show the graph
    plt.show()
def generate_data():
    radar_plots()
    ip = 0  # participant number
    df = pd.DataFrame(columns=['participant', 'group', 'game', 'trust_score', 'reward'])
    df_means = pd.DataFrame(columns=['_','participant', 'group', 'trust_score','d_trust','reward'])

    # Generate data for coactive agent  ----------------
    n_participants = 50
    trust_mu0 = 4.5; trust_s0 = 2
    trust_mu1 = 4; trust_s1 = 1.5
    trust_mu2 = 2; trust_s2 = 1
    coactive_Ti = np.array([np.random.normal(trust_mu0, trust_s0, n_participants),
                   np.random.normal(trust_mu1, trust_s1, n_participants),
                   np.random.normal(trust_mu2, trust_s2, n_participants)])
    # coactive_dTi = [coactive_Ti[1] - coactive_Ti[0],
    #                 coactive_Ti[2] - coactive_Ti[1]]
    reward_mu0 = 15; reward_s0 = 7
    reward_mu1 = 13; reward_s1 = 7
    reward_mu2 = 12; reward_s2 = 7
    coactive_Ri = np.array([np.random.normal(reward_mu0, reward_s0, n_participants),
                    np.random.normal(reward_mu1, reward_s1, n_participants),
                    np.random.normal(reward_mu2, reward_s2, n_participants)])

    group = 'coactive'
    rawT = coactive_Ti
    rawR = coactive_Ri
    for iP in range(n_participants):
        # Calc means
        mean_reward = rawR[:, iP].mean()
        mean_trust = rawT[:, iP].mean()
        mean_d_trust = np.mean([rawT[1, iP] - rawT[0, iP], rawT[2, iP] - rawT[1, iP]])
        row_mean = {'_':'','participant': iP, 'group': group,
                    'trust_score': mean_trust, 'd_trust': mean_d_trust, 'reward': mean_reward}
        df_means = df_means.append(row_mean, ignore_index=True)

        # Calc individual games
        for iG in range(3):
            row = {'participant': ip, 'group': group, 'game': iG + 1,
                   'trust_score': rawT[iG][iP], 'reward': rawR[iG][iP]}
            df = df.append(row, ignore_index=True)
            # df = pd.concat(df, pd.DataFrame(row), keys=row.key(),axis = 0)
        ip += 1
    # Generate data for directive agent ----------------

    trust_mu0 = 4;  trust_s0 = 2
    trust_mu1 = 5; trust_s1 = 1.5
    trust_mu2 = 7; trust_s2 = 1
    directive_Ti = np.array([np.random.normal(trust_mu0, trust_s0, n_participants),
                    np.random.normal(trust_mu1, trust_s1, n_participants),
                    np.random.normal(trust_mu2, trust_s2, n_participants)])
    # directive_dTi = [directive_Ti[1] - directive_Ti[0],
    #                  directive_Ti[2] - directive_Ti[1]]

    reward_mu0 = 20; reward_s0 = 5
    reward_mu1 = 19; reward_s1 = 5
    reward_mu2 = 16; reward_s2 = 5
    directive_Ri = np.array([np.random.normal(reward_mu0, reward_s0, n_participants),
                    np.random.normal(reward_mu1, reward_s1, n_participants),
                    np.random.normal(reward_mu2, reward_s2, n_participants)])

    group = 'directive'
    rawT = directive_Ti
    rawR = directive_Ri
    for iP in range(n_participants):
        # Calc means
        mean_reward = np.mean(rawR[:,iP])
        mean_trust = np.mean(rawT[:,iP])
        mean_d_trust =  np.mean([rawT[1,iP]-rawT[0,iP],rawT[2,iP]-rawT[1,iP]])
        row_mean = {'_':'','participant': iP+n_participants, 'group': group,
                    'trust_score': mean_trust, 'd_trust': mean_d_trust, 'reward': mean_reward}
        df_means = df_means.append(row_mean, ignore_index=True)

        # Calc individual games
        for iG in range(3):
            row = {'participant': ip, 'group': group, 'game': iG+1, 'trust_score': rawT[iG][iP], 'reward': rawR[iG][iP]}
            df = df.append(row, ignore_index=True)
            # df = pd.concat(df, pd.DataFrame(row),keys=row.key())
        ip += 1
    return df, df_means

def main():
    BP_fmt_kwargs = {'notch':True,'dodge':True} #'medianprops':{"color": "r"},


    # Example
    # data = sns.load_dataset('tips')
    # BP = sns.boxplot(x=data['day'], y=data['total_bill'], hue=data['sex'])

    # Simulated Data ----------------
    data, data_means = generate_data()
    # my_pal = {group: "lightgray" if group == "coactive" else "tab:blue" for group in data.group.unique()}


    # TRUST ###################################
    fig, axs = plt.subplots(1, 3,constrained_layout=True)  # figsize=(10,5)
    my_pal = {"coactive": "lightgray", "directive": "tab:blue"}
    # Plot trust per game ----------------
    BP_trust_series = sns.boxplot(ax= axs[0],x=data['game'], y=data['trust_score'], hue=data['group'],palette=my_pal,**BP_fmt_kwargs)
    BP_trust_series.legend_.set_title(None) # remove legend title
    BP_trust_series.set_ylabel('Trust Score')
    BP_trust_series.set_xlabel('Game Number')

    # Plot trust means ----
    BP_trust_measn = sns.boxplot(ax=axs[1],  y=data_means['trust_score'], hue=data_means['group'],palette=my_pal,**BP_fmt_kwargs)
    BP_trust_measn.legend_.set_title(None)  # remove legend title
    BP_trust_measn.set_ylabel('Average Trust Score')


    # Plot trust means ----
    BP_trust_measn = sns.boxplot(ax=axs[2],  y=data_means['d_trust'], hue=data_means['group'],palette=my_pal,**BP_fmt_kwargs)
    BP_trust_measn.legend_.set_title(None)  # remove legend title
    BP_trust_measn.set_ylabel('Average Trust Change ($\Delta$ Trust)')





    # REWARDS ###################################
    fig, axs = plt.subplots(1, 2,constrained_layout=True)  # figsize=(10,5)
    # my_pal = {"coactive": "lightgray", "directive": "tab:green"}
    # Plot reward per game ----------------
    BP_trust_series = sns.boxplot(ax=axs[0], x=data['game'], y=data['reward'], hue=data['group'],palette=my_pal, **BP_fmt_kwargs)
    BP_trust_series.legend_.set_title(None)  # remove legend title
    BP_trust_series.set_ylabel('Reward')
    BP_trust_series.set_xlabel('Game Number')

    # Plot reward means ----
    BP_trust_measn = sns.boxplot(ax=axs[1], y=data_means['reward'], hue=data_means['group'],palette=my_pal, **BP_fmt_kwargs)
    BP_trust_measn.legend_.set_title(None)  # remove legend title
    BP_trust_measn.set_ylabel('Mean Reward')



    # Create a grouped boxplot


    plt.show()

if __name__ == "__main__":
    main()