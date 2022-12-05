import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sim_analysis(sim_list, NUM_SIM, NUM_SEEDS=1, SIM_DAYS=10):
    NUM_MM_AGENTS = len(sim_list[0].mm_agent_list)

    # eps density
    plot_names = ['bid spread', 'ask spread', 'hedge ratio']
    mm_names = [sim_list[0].mm_agent_list[i].name for i in range(NUM_MM_AGENTS)]
    mm_eps_dict = {}
    for i in range(NUM_MM_AGENTS):
        mm_name = mm_names[i]
        eps_dict = {k: [] for k in plot_names}

        for j in range(NUM_SIM*NUM_SEEDS):
            if (len(sim_list[j].mm_agent_list[i].eps_bid_history)>26*SIM_DAYS or
                        len(sim_list[j].mm_agent_list[i].eps_ask_history)>26*SIM_DAYS or
                        len(sim_list[j].mm_agent_list[i].hedge_coeffs)>26*SIM_DAYS):
                    break
            eps_dict['bid spread'] += list(sim_list[j].mm_agent_list[i].eps_bid_history)
            eps_dict['ask spread'] += list(sim_list[j].mm_agent_list[i].eps_ask_history)
            eps_dict['hedge ratio'] += list(sim_list[j].mm_agent_list[i].hedge_coeffs)

        mm_eps_dict[mm_name] = eps_dict

    mm_names_relevant = np.array([(name if not ('random' in name.lower() or 'persistent' in name.lower()) else 'NA')
                        for name in mm_names])
    mm_names_relevant = mm_names_relevant[mm_names_relevant!='NA'].tolist()
    mm_eps_dict_relevant = pd.DataFrame(mm_eps_dict)[mm_names_relevant].T.to_dict()

    for plot_name in plot_names:
        df_eps = pd.DataFrame(mm_eps_dict_relevant[plot_name])
        plt.figure(figsize=(8, 6))
        sns.set_style('whitegrid')
        try:
            sns.kdeplot(data=df_eps)
        except:
            pass
        plt.title(plot_name)
        plt.show()

    NUM_MM_AGENTS = len(sim_list[0].mm_agent_list)

    # pnl
    plot_names = ['spread pnl', 'inventory pnl', 'total pnl', 'inventory']
    table_names = ['hedge cost per time-step', 'spread pnl per time-step', 'inventory pnl per time-step',
    'total pnl per time-step','Reward per time-step']
    temp_names = []#['negative inventory pnl', 'inventory punished total pnl']
    stat_names = plot_names + table_names + temp_names
    mm_names = [sim_list[0].mm_agent_list[i].name for i in range(NUM_MM_AGENTS)]
    mm_pnl_dict = {}
    for i in range(NUM_MM_AGENTS):
        mm_name = sim_list[0].mm_agent_list[i].name
        pnl_dict = {k: [] for k in stat_names}

        for j in range(NUM_SIM*NUM_SEEDS):
            if (len(sim_list[j].mm_agent_list[i].pnl_spread)>26*SIM_DAYS or
                    len(sim_list[j].mm_agent_list[i].hedge_cost)>26*SIM_DAYS or
                    len(sim_list[j].mm_agent_list[i].pnl_inventory)>26*SIM_DAYS or
                    len(sim_list[j].mm_agent_list[i].inventory_history)>26*SIM_DAYS):
                break
            pnl_dict['hedge cost per time-step'].append(-pd.Series(sim_list[j].mm_agent_list[i].hedge_cost))
            pnl_dict['spread pnl per time-step'].append(pd.Series(sim_list[j].mm_agent_list[i].pnl_spread))
            pnl_dict['inventory pnl per time-step'].append(pd.Series(sim_list[j].mm_agent_list[i].pnl_inventory))
            pnl_dict['total pnl per time-step'].append(pnl_dict['spread pnl per time-step'][-1] +
                                                    pnl_dict['inventory pnl per time-step'][-1] +
                                                    pnl_dict['hedge cost per time-step'][-1])
            pnl_dict['Reward per time-step'].append(pnl_dict['spread pnl per time-step'][-1] -0.5*
                                                    np.abs(pnl_dict['inventory pnl per time-step'][-1]))

            pnl_dict['inventory'].append(pd.Series(sim_list[j].mm_agent_list[i].inventory_history))
            pnl_dict['spread pnl'].append(pnl_dict['spread pnl per time-step'][-1].cumsum())
            pnl_dict['inventory pnl'].append(pnl_dict['inventory pnl per time-step'][-1].cumsum())
            pnl_dict['total pnl'].append(pnl_dict['total pnl per time-step'][-1].cumsum())

        mm_pnl_dict[mm_name] = pnl_dict

    mm_pnl_table_dict = pd.DataFrame(mm_pnl_dict).T[table_names].to_dict()
    mm_pnl_df = {}
    for tbl_name in table_names:
        mm_pnl_df[tbl_name] = {mm_name: np.mean(mm_pnl_table_dict[tbl_name][mm_name]) for mm_name in mm_names}

    mm_pnl_df = pd.DataFrame(mm_pnl_df).T
    print(mm_pnl_df)

    mm_pnl_plot_dict = pd.DataFrame(mm_pnl_dict).T[plot_names + temp_names].to_dict()

    for plot_name in plot_names + temp_names:
        plot_data = {}
        pnl_dict = mm_pnl_plot_dict[plot_name]
        for mm_name in mm_names:
            plot_data[mm_name] = np.row_stack(pnl_dict[mm_name]).mean(axis=0)
        plot_data = pd.DataFrame(plot_data)
        plt.figure(figsize=(12, 8))
        sns.set_style('whitegrid')
        # plt.plot(plot_data, label=plot_data.columns)
        sns.lineplot(data=plot_data)
        plt.legend()
        plt.title(plot_name)
        plt.show()

    RL_NAME = 'RL Agent'
    bid_eps = np.array(mm_eps_dict[RL_NAME]['bid spread']).reshape(-1)
    ask_eps = np.array(mm_eps_dict[RL_NAME]['ask spread']).reshape(-1)
    inventory = np.array(mm_pnl_dict[RL_NAME]['inventory']).reshape(-1)
    plt.figure(figsize=(12, 8))
    sns.set_style('whitegrid')
    plt.scatter(inventory, bid_eps, label='bid eps')
    plt.scatter(inventory, ask_eps, label='ask eps')
    plt.xlabel('inventory')
    plt.legend()
    plt.title('RL eps')
    plt.show()

    pd.DataFrame({'inventory': np.mean(inventory.reshape(NUM_SIM * NUM_SEEDS, -1), axis=0),
                  'spread_bid': np.mean(bid_eps.reshape(NUM_SIM * NUM_SEEDS, -1), axis=0),
                  'spread_ask': np.mean(ask_eps.reshape(NUM_SIM * NUM_SEEDS, -1), axis=0)}).plot(
        secondary_y=['inventory'], figsize=(12, 8))
    plt.show()

    return mm_eps_dict, mm_pnl_dict
