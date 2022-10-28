import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import random


def viz_scenario(lane_centerlines, agents_trajs_skewed, agent_gt, agent_prediction, save_addr, rot, orig, n_trajectory, save=True,
                 x_radius=50, y_radius=50, legend=True, show_agents=True, ruler=True):
    # fig, ax = plt.subplots()
    if legend:
        plt.clf()
    x_min, x_max = agents_trajs_skewed[0][19][0] - x_radius, agents_trajs_skewed[0][19][0] + x_radius
    y_min, y_max = agents_trajs_skewed[0][19][1] - y_radius, agents_trajs_skewed[0][19][1] + y_radius
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    color_dict = {"AGENT_HISTORY": "#a6961b", "AGENT_GT": "#006B73",
                  "AGENT_PRED": "#DA4749", "AGENT_PRED1": "#FF7300", "AGENT_PRED2": "#FFA200",
                  "OTHERS": "#929eea", "AV": "#007672", "AGENT_HISTORY_ORIG": "#804600", "AGENT_PRED_ORIG": "#DE00CB",
                  "Road_color": "#A5A5A3"}
    # plot map
    for lane_cl in lane_centerlines:
        lane_cl = np.matmul(rot.T, lane_cl.T).T + orig.reshape(-1, 2)
        plt.plot(
            lane_cl[:, 0],
            lane_cl[:, 1],
            "-",
            color=color_dict["Road_color"],
            alpha=1,
            linewidth=1,
            zorder=0,
        )
    plt.axis("off")

    # plot vehicles other than agent
    for i in range(1, len(agents_trajs_skewed)):
        current_data = agents_trajs_skewed[i][:50]
        if len(current_data) > 5 and np.sqrt(((current_data[-1] - current_data[0]) ** 2).sum()) >= 6 and show_agents:
            if i != 1:
                plt.plot(
                    current_data[:, 0],
                    current_data[:, 1],
                    "-",
                    color=color_dict["OTHERS"],
                    alpha=1,
                    linewidth=1.5,
                    zorder=0,
                )
            else:
                plt.plot(
                    current_data[:, 0],
                    current_data[:, 1],
                    "-",
                    color=color_dict["OTHERS"],
                    alpha=1,
                    linewidth=1.5,
                    zorder=0,
                    label='Other Vehicles'
                )
            m, b = np.polyfit(current_data[-5:, 0], current_data[-5:, 1], 1)
            plt.arrow(current_data[-2, 0], current_data[-2, 1],
                      (current_data[-1, 0] - current_data[-5, 0]) / np.abs(
                          current_data[-1, 0] - current_data[-5, 0]) / np.sqrt(1 + m ** 2),
                      m * (current_data[-1, 0] - current_data[-5, 0]) / np.abs(
                          current_data[-1, 0] - current_data[-5, 0]) / np.sqrt(1 + m ** 2)
                      , color=color_dict["OTHERS"], width=0.3)

    # plot agent

    agent_history = agents_trajs_skewed[0][:20, :]
    plt.plot(
        agent_history[:, 0],
        agent_history[:, 1],
        "-",
        color=color_dict["AGENT_HISTORY"],
        label='Observation',
        alpha=1,
        linewidth=2,
        zorder=0,
    )
    plt.plot(
        agent_gt[:, 0],
        agent_gt[:, 1],
        "-",
        color=color_dict["AGENT_GT"],
        label='Ground Truth',
        alpha=1,
        linewidth=2,
        zorder=0,
    )
    plt.arrow(agent_gt[-2, 0], agent_gt[-2, 1], agent_gt[-1, 0] - agent_gt[-2, 0],
              agent_gt[-1, 1] - agent_gt[-2, 1], color=color_dict["AGENT_GT"], width=0.2)
    circle_gt = plt.Circle((agent_gt[-1, 0], agent_gt[-1, 1]), radius=0.2, color='#069AF3')
    plt.gca().add_patch(circle_gt)

    color_map = mpl.cm.get_cmap('hsv', n_trajectory+1)
    max_cls = max([x[1] for x in agent_prediction])
    for idx, (reg, cls) in enumerate(agent_prediction):
        if idx < n_trajectory:
            label = "%.1f (%.4f)" % (1.0, cls) if idx == 0 else "%.4f" % round(cls/max_cls, 4)
            plt.plot(
                reg[:, 0],
                reg[:, 1],
                "-",
                color=color_map(idx),
                label=label,
                alpha=1,
                linewidth=2,
                zorder=0,
            )
            plt.arrow(reg[-2, 0], reg[-2, 1], reg[-1, 0] - reg[-2, 0],
                      reg[-1, 1] - reg[-2, 1], color=color_map(idx), width=0.2)
            circle_pred = plt.Circle((reg[-1, 0],reg[-1, 1]), radius=0.2, color=color_map(idx))
            plt.gca().add_patch(circle_pred)

    if legend:
        plt.legend(loc="lower right", prop={"size": "small"})
    if ruler:
        plt.plot(np.array([orig[0] - 0.9 * x_radius, orig[0] - 0.7 * x_radius]),
                 np.array([orig[1] - 0.9 * y_radius, orig[1] - 0.9 * y_radius]), color="black", linewidth=3)
        plt.plot(np.array([orig[0] - 0.91 * x_radius, orig[0] - 0.91 * x_radius]),
                 np.array([orig[1] - 0.88 * y_radius, orig[1] - 0.92 * y_radius]), color="black", linewidth=2)
        plt.plot(np.array([orig[0] - 0.7 * x_radius, orig[0] - 0.7 * x_radius]),
                 np.array([orig[1] - 0.88 * y_radius, orig[1] - 0.92 * y_radius]), color="black", linewidth=2)

        plt.text(orig[0] - 0.87 * x_radius, orig[1] - 0.85 * y_radius, str(int(x_radius * 0.2)) + "m")

    if save:
        if not os.path.exists(save_addr[: save_addr.rfind('/')]):
            os.makedirs(save_addr[: save_addr.rfind('/')])
        plt.savefig(save_addr + ".jpg", bbox_inches='tight')
