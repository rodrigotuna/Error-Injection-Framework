import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from dm_test import dm_test
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error, mean_squared_error

params = {
    'mag_mean' : [1,3,5],
    'mag_std' : [0.1, 0.2, 0.5],
    'freq_mean' : [0.01, 0.05, 0.1, 0.15, 0.20, 0.25],
    'freq_std' : [0.1, 0.2, 0.5]
}

with open(f"pred_plots_exp/metric_lstm.json", 'r') as infile:
    lstm_results = json.load(infile)

with open(f"pred_plots_exp/metric_clstm.json", 'r') as infile:
    clstm_results = json.load(infile)


mase_lstm = {}
mase_clstm = {}

rmse_lstm = {}
rmse_clstm = {}

wins_by_series = { i : {"wins" : 0, "draws" : 0, "losses" : 0} for i in range(1,21)}

wins_by_vars = {}

wins_lstm = 0
wins_clstm = 0
draws = 0

for size in range(1,6,2):
    for num in [0.01, 0.05, 0.1, 0.15, 0.20, 0.25]:
        for mag_p in [0.1, 0.2, 0.5]:
            for loc_p in [0.1, 0.2, 0.5]:
                wins_var_lstm = 0
                wins_var_clstm = 0
                draws_var = 0

                clstm_array = []
                lstm_array = []

                rmse_lstm[f"{size}-{mag_p}-{num}-{loc_p}"] = {}
                rmse_clstm[f"{size}-{mag_p}-{num}-{loc_p}"] = {}

                for unique_id in range(1,21):
                    results = lstm_results[str(unique_id)][f"{size}-{mag_p}-{num}-{loc_p}"]
                    mase = mean_absolute_scaled_error(y_true=np.array(results["y_true"]), y_pred=np.array(results["y_pred"]), y_train=np.array(results["y_train"]))
                    lstm_array.append(mase)

                    rmse_l = np.sqrt(mean_squared_error(y_true=np.array(results["y_true"]), y_pred=np.array(results["y_pred"])))
                    rmse_lstm[f"{size}-{mag_p}-{num}-{loc_p}"][unique_id] = rmse_l
                    if(f"{size}-{mag_p}-{num}-{loc_p}" not in clstm_results[str(unique_id)].keys()):
                        continue

                    results = clstm_results[str(unique_id)][f"{size}-{mag_p}-{num}-{loc_p}"]
                    mase = mean_absolute_scaled_error(y_true=np.array(results["y_true"]), y_pred=np.array(results["y_pred"]), y_train=np.array(results["y_train"]))
                    clstm_array.append(mase)

                    rmse_c = np.sqrt(mean_squared_error(y_true=np.array(results["y_true"]), y_pred=np.array(results["y_pred"])))
                    rmse_clstm[f"{size}-{mag_p}-{num}-{loc_p}"][unique_id] = rmse_c

                    (DM,_) = dm_test(np.array(results["y_true"]), np.array(clstm_results[str(unique_id)][f"{size}-{mag_p}-{num}-{loc_p}"]["y_pred"]), np.array(lstm_results[str(unique_id)][f"{size}-{mag_p}-{num}-{loc_p}"]["y_pred"]))
                    if DM < -1.96 or DM > 1.96:
                        if DM < -1.96:
                            wins_by_series[unique_id]["wins"]+=1
                            wins_clstm += 1
                            wins_var_clstm += 1
                        elif DM > 1.96:
                            wins_by_series[unique_id]["losses"]+=1
                            wins_var_lstm += 1
                            wins_lstm += 1
                    else:
                         draws_var += 1
                         wins_by_series[unique_id]["draws"]+=1
                         draws += 1

                wins_by_vars[f"{size}-{mag_p}-{num}-{loc_p}"] = {"wins": wins_var_clstm, "draws" : draws_var,  "losses" : wins_var_lstm}
                mase_lstm[f"{size}-{mag_p}-{num}-{loc_p}"] = np.mean(lstm_array)
                if len(clstm_array) == 0: continue   
                mase_clstm[f"{size}-{mag_p}-{num}-{loc_p}"] = np.mean(clstm_array)


with open(f"pred_plots_exp/mase_lstm.json", 'w') as outfile:
        json.dump(mase_lstm, outfile, indent=2)

with open(f"pred_plots_exp/mase_clstm.json", 'w') as outfile:
        json.dump(mase_clstm, outfile, indent=2)

with open(f"pred_plots_exp/rmse_lstm.json", 'w') as outfile:
        json.dump(rmse_lstm, outfile, indent=2)

with open(f"pred_plots_exp/rmse_clstm.json", 'w') as outfile:
        json.dump(rmse_clstm, outfile, indent=2)

with open(f"pred_plots_exp/wins_vars.json", 'w') as outfile:
        json.dump(wins_by_vars, outfile, indent=2)

with open(f"pred_plots_exp/wins_series.json", 'w') as outfile:
        json.dump(wins_by_series, outfile, indent=2)


print(f"Wins LSTM: {wins_lstm}\nWins CLSTM: {wins_clstm}\nDraws: {draws}" )

param_list = list(params.keys())

for i in range(0, len(param_list)):
    for j in range(i+1, len(param_list)):
        plot_grid = []
        ind = [0,1,2,3]
        ind.remove(i)
        ind.remove(j)
        for a in params[param_list[i]]:
            plot_line = []
            for b in params[param_list[j]]:
                lstm_array = []
                clstm_array = []

                for x in params[param_list[ind[0]]]:
                    for y in params[param_list[ind[1]]]:
                        param_values = [f"1{a}",f"2{b}",f"3{x}",f"4{y}"]
                        param_pos = {f"1{a}" : i, f"2{b}" : j, f"3{x}" : ind[0], f"4{y}": ind[1]}
                        param_values.sort(key=(lambda e : param_pos[e]))
                        param_values = list(map((lambda x : x[1:]), param_values))
                        lstm_array.append(mase_lstm[f"{param_values[0]}-{param_values[1]}-{param_values[2]}-{param_values[3]}"])
                        if(f"{param_values[0]}-{param_values[1]}-{param_values[2]}-{param_values[3]}" not in mase_clstm.keys()): continue
                        clstm_array.append(mase_clstm[f"{param_values[0]}-{param_values[1]}-{param_values[2]}-{param_values[3]}"])

                plot_line.append(np.mean(lstm_array) - np.mean(clstm_array))
            plot_grid.append(plot_line)

        ax = sns.heatmap(plot_grid, xticklabels=params[param_list[j]], yticklabels=params[param_list[i]], cmap="RdYlGn")
        ax.set_ylabel(param_list[i])
        ax.set_xlabel(param_list[j])
        ax.set_title('Average MASE difference')
        plt.savefig(f"result_plots/heatmap-{param_list[i]}-{param_list[j]}.pdf")
        plt.clf()


def bar_plot(param, wins, draws, losses):
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(wins))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    # Make the plot
    plt.bar(br1, wins, color ='g', width = barWidth,
            edgecolor ='grey', label ='wins')
    plt.bar(br2, draws, color ='y', width = barWidth,
            edgecolor ='grey', label ='draws')
    plt.bar(br3, losses, color ='r', width = barWidth,
            edgecolor ='grey', label ='losses')
    
    # Adding Xticks
    plt.xlabel(param, fontweight ='bold', fontsize = 15)
    plt.ylabel('Wins/Draws/Losses', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(wins))],
            params)
    
    plt.legend()
    plt.savefig(f"result_plots/{param}_bar.pdf")

langs = ['Wins', 'Draws', 'Losses']
students = [1467, 557, 1216]
plt.bar(langs,students,tick_label=langs)
plt.savefig(f"./pred_plots_exp/mariano.pdf")
plt.clf()

with open(f"./pred_plots_exp/wins_vars.json","r") as f:
    data = json.load(f)

params = []
wins = []
draws = []
losses = []
for size in range(1,6,2):
    params.append(size)
    winsv = 0
    drawsv = 0
    lossesv = 0
    for num in [0.01, 0.05, 0.1, 0.15, 0.20, 0.25]:
        for mag_p in [0.1, 0.2, 0.5]:
            for loc_p in [0.1, 0.2, 0.5]:
                winsv += data[f"{size}-{mag_p}-{num}-{loc_p}"]["wins"]
                drawsv += data[f"{size}-{mag_p}-{num}-{loc_p}"]["draws"]
                lossesv += data[f"{size}-{mag_p}-{num}-{loc_p}"]["losses"]
    wins.append(winsv)
    draws.append(drawsv)
    losses.append(lossesv)

bar_plot("mag_mean", wins, draws, losses)

params = []
wins = []
draws = []
losses = []
for num in [0.01, 0.05, 0.1, 0.15, 0.20, 0.25]:
    params.append(num)
    winsv = 0
    drawsv = 0
    lossesv = 0
    for size in range(1,6,2):
        for mag_p in [0.1, 0.2, 0.5]:
            for loc_p in [0.1, 0.2, 0.5]:
                winsv += data[f"{size}-{mag_p}-{num}-{loc_p}"]["wins"]
                drawsv += data[f"{size}-{mag_p}-{num}-{loc_p}"]["draws"]
                lossesv += data[f"{size}-{mag_p}-{num}-{loc_p}"]["losses"]
    wins.append(winsv)
    draws.append(drawsv)
    losses.append(lossesv)

bar_plot("freq_mean", wins, draws, losses)


params = []
wins = []
draws = []
losses = []
for mag_p in [0.1, 0.2, 0.5]:
    params.append(mag_p)
    winsv = 0
    drawsv = 0
    lossesv = 0
    for size in range(1,6,2):
        for num in [0.01, 0.05, 0.1, 0.15, 0.20, 0.25]:
            for loc_p in [0.1, 0.2, 0.5]:
                winsv += data[f"{size}-{mag_p}-{num}-{loc_p}"]["wins"]
                drawsv += data[f"{size}-{mag_p}-{num}-{loc_p}"]["draws"]
                lossesv += data[f"{size}-{mag_p}-{num}-{loc_p}"]["losses"]
    wins.append(winsv)
    draws.append(drawsv)
    losses.append(lossesv)

bar_plot("mag_std", wins, draws, losses)


params = []
wins = []
draws = []
losses = []
for loc_p in [0.1, 0.2, 0.5]:
    params.append(loc_p)
    winsv = 0
    drawsv = 0
    lossesv = 0
    for size in range(1,6,2):
        for num in [0.01, 0.05, 0.1, 0.15, 0.20, 0.25]:
            for mag_p in [0.1, 0.2, 0.5]:
                winsv += data[f"{size}-{mag_p}-{num}-{loc_p}"]["wins"]
                drawsv += data[f"{size}-{mag_p}-{num}-{loc_p}"]["draws"]
                lossesv += data[f"{size}-{mag_p}-{num}-{loc_p}"]["losses"]
    wins.append(winsv)
    draws.append(drawsv)
    losses.append(lossesv)

bar_plot("freq_std", wins, draws, losses)

def indiv(model):
    with open(f"./pred_plots_exp/mase_{model}.json","r") as f:
        data = json.load(f)

    params = {
        'mag_mean' : [1,3,5],
        'mag_std' : [0.1, 0.2, 0.5],
        'freq_mean' : [0.01, 0.05, 0.1, 0.15, 0.20, 0.25],
        'freq_std' : [0.1, 0.2, 0.5]
    }

    param_list = list(params.keys())
    mase_list = list(data.values())

    print(f"Average MASE = {np.mean(mase_list)}")

    for i in range(0, len(param_list)):
        for j in range(i+1, len(param_list)):
            plot_grid = []
            ind = [0,1,2,3]
            ind.remove(i)
            ind.remove(j)
            for a in params[param_list[i]]:
                plot_line = []
                for b in params[param_list[j]]:
                    model_array = []

                    for x in params[param_list[ind[0]]]:
                        for y in params[param_list[ind[1]]]:
                            param_values = [f"1{a}",f"2{b}",f"3{x}",f"4{y}"]
                            param_pos = {f"1{a}" : i, f"2{b}" : j, f"3{x}" : ind[0], f"4{y}": ind[1]}
                            param_values.sort(key=(lambda e : param_pos[e]))
                            param_values = list(map((lambda x : x[1:]), param_values))
                            model_array.append(data[f"{param_values[0]}-{param_values[1]}-{param_values[2]}-{param_values[3]}"])

                    plot_line.append(np.mean(model_array))
                plot_grid.append(plot_line)

            ax = sns.heatmap(plot_grid, xticklabels=params[param_list[j]], yticklabels=params[param_list[i]], cmap=sns.cm.rocket_r)
            ax.set_ylabel(param_list[i])
            ax.set_xlabel(param_list[j])
            ax.set_title(f'Average MASE for {model}')
            plt.savefig(f"result_plots/{model}-heatmap-{param_list[i]}-{param_list[j]}.pdf")
            plt.clf()

indiv("lstm")
indiv("clstm")