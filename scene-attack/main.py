import numpy as np
import argparse
import datetime as dt
import pickle
import os
from tqdm import tqdm
import copy

import scenario_attacker
import blackboxOptimizers


parser = argparse.ArgumentParser(description='main arguments')
parser.add_argument("--save_addr", type=str, default="")                        # directory for saveing visualizations
parser.add_argument("--n_scenarios", type=int, default=20)              
parser.add_argument("--model_name", type=str, default="LaneGCN")                # options: ["LaneGCN", "MPC", "DATF", "WIMP"]
parser.add_argument('--start_scenario', type = int, default=0)                 
parser.add_argument('--optimization_type', type=str, default="")                # options: ["baysian", "parzen", "evolution"]
parser.add_argument('--transfer_from', type=str, default="")                    
parser.add_argument('--transfer', dest='transfer', action='store_true')
parser.add_argument('--n_trajectory', type=int, default=1)
parser.set_defaults(transfer=False)
args = parser.parse_args()


default_params = {"smooth-turn": {"attack_power": 0, "pow": 3, "border": 5},
                  "double-turn": {"attack_power": 0, "pow": 3, "l": 10, "border": 5},
                  "ripple-road": {"attack_power": 0, "l": 60, "border": 5}}


def main_test():
    if args.n_trajectory not in range(1, 7):
        raise Exception(f'Can not draw {args.n_trajectory} trajectories. Please type a number between 1 and 6')
    
    augmentor = scenario_attacker.SceneAugmentor(models_name=args.model_name, render=True)
    augmentor.attack(args.n_scenarios,
                    default_params, 
                    save_addr=f"./scenario-images/{args.save_addr}", 
                    n_trajectory=args.n_trajectory) 

def main():
    if args.model_name not in ["LaneGCN", "MPC", "DATF", "WIMP"]:
        raise Exception('Attacking model not found. Please change the attacking model.')
    if args.n_trajectory not in range(1, 7):
        raise Exception(f'Can not draw {args.n_trajectory} trajectories. Please type a number between 1 and 6')
    save_dir = os.path.join("./scenario-images", args.save_addr)
    global augmentor, OR
    OR = [0, 0]
    augmentor = scenario_attacker.SceneAugmentor(models_name=args.model_name, render = not args.save_addr == "")
    if(not args.optimization_type == ''):
        black_box_attack(args)
        return 0
    if(not args.transfer_from == ''):
        transfer(args)
        return 0
    if(args.transfer):
        transfer_data = []
    attack_functions = ["smooth-turn", "double-turn", "ripple-road"]
    results = {"smooth-turn": {"HOR": 0, "SOR": 0, "ADE": 0,"FDE": 0, "minADE": 0,"minFDE": 0},
               "double-turn": {"HOR": 0, "SOR": 0, "ADE": 0,"FDE": 0, "minADE": 0,"minFDE": 0},
               "ripple-road": {"HOR": 0, "SOR": 0, "ADE": 0,"FDE": 0, "minADE": 0,"minFDE": 0},
               "original": {"HOR": 0, "SOR": 0, "ADE": 0,"FDE": 0, "minADE": 0,"minFDE": 0},
               "all": {"HOR": 0, "SOR": 0, "ADE": 0,"FDE": 0, "minADE": 0,"minFDE": 0}}
    for scenario in tqdm(range(args.n_scenarios)):
        all_SOR, all_HOR = 0, 0
        all_ADE=all_FDE=all_minADE=all_minFDE = 0
        best_params = default_params
        save_addr = save_dir + '/' + "original/" + str(scenario) + "/0"
        original_offroad_rate, original_displ_error = augmentor.attack(scenario, default_params, save_addr, args.n_trajectory)
        results["original"]["SOR"] += original_offroad_rate[0] * 100 / args.n_scenarios
        results["original"]["HOR"] += original_offroad_rate[1] * 100 / args.n_scenarios
        results["original"]["ADE"] += original_displ_error[0] / args.n_scenarios
        results["original"]["FDE"] += original_displ_error[1] / args.n_scenarios
        results["original"]["minADE"] += original_displ_error[2] / args.n_scenarios
        results["original"]["minFDE"] += original_displ_error[3] / args.n_scenarios
        for attack_type in attack_functions:
            attack_SOR, attack_HOR = 0, 0
            attack_ADE=attack_FDE=attack_minADE=attack_minFDE = 0
            params = copy.deepcopy(default_params)
            for attack_power in range(-9, 10):
                if attack_power == 0:
                    continue
                params[attack_type]["attack_power"] = attack_power
                save_addr = save_dir + '/' + attack_type + '/' + str(scenario) + '/' + str(attack_power)
                offroad_rate, displ_error =  augmentor.attack(scenario, params, save_addr, args.n_trajectory) 
                SOR, HOR = offroad_rate
                ADE, FDE, minADE, minFDE = displ_error
                if(SOR > all_SOR):
                    best_params = copy.deepcopy(params)
                attack_SOR = max(attack_SOR, SOR)
                attack_HOR = max(attack_HOR, HOR)
                attack_ADE = max(attack_ADE, ADE)
                attack_FDE = max(attack_FDE, FDE)
                attack_minADE = max(attack_minADE, minADE)
                attack_minFDE = max(attack_minFDE, minFDE)
                all_SOR = max(all_SOR, SOR)
                all_HOR = max(all_HOR, HOR)
                all_ADE = max(all_ADE, ADE)
                all_FDE = max(all_FDE, FDE)
                all_minADE = max(all_minADE, minADE)
                all_minFDE = max(all_minFDE, minFDE)
            results[attack_type]["SOR"] += attack_SOR * 100 / args.n_scenarios
            results[attack_type]["HOR"] += attack_HOR * 100 / args.n_scenarios
            results[attack_type]["ADE"] += attack_ADE / args.n_scenarios
            results[attack_type]["FDE"] += attack_FDE / args.n_scenarios
            results[attack_type]["minADE"] += attack_minADE / args.n_scenarios
            results[attack_type]["minFDE"] += attack_minFDE / args.n_scenarios
        results["all"]["SOR"] += all_SOR * 100 / args.n_scenarios
        results["all"]["HOR"] += all_HOR * 100 / args.n_scenarios
        results["all"]["ADE"] += all_ADE / args.n_scenarios
        results["all"]["FDE"] += all_FDE / args.n_scenarios
        results["all"]["minADE"] += all_minADE / args.n_scenarios
        results["all"]["minFDE"] += all_minFDE / args.n_scenarios
        if(args.transfer):
            transfer_data.append((scenario, best_params, all_SOR))
    if(args.transfer):
        write(transfer_data, 'transfer-attack-files/' + args.model_name + '-' + str(args.n_scenarios))
        
    result = "Model: " + args.model_name + "\n"
    result += "\t\t\t Original \t Smooth-turn \t Double-turn \t Ripple-road \t    All\n"
    result += "(SOR/HOR)\t" + "\t   {}/{} \t\t    {}/{} \t    {}/{} \t    {}/{} \t    {}/{}\n".format(
        round(results["original"]["SOR"]), round(results["original"]["HOR"]),
        round(results["smooth-turn"]["SOR"]), round(results["smooth-turn"]["HOR"]),
        round(results["double-turn"]["SOR"]), round(results["double-turn"]["HOR"]),
        round(results["ripple-road"]["SOR"]), round(results["ripple-road"]["HOR"]),
        round(results["all"]["SOR"]), round(results["all"]["HOR"]))
    result += "(ADE/FDE)\t" + "\t{}/{} \t {}/{} \t {}/{} \t {}/{} \t {}/{}\n".format(
        round(results["original"]["ADE"], 2), round(results["original"]["FDE"], 2),
        round(results["smooth-turn"]["ADE"], 2), round(results["smooth-turn"]["FDE"], 2),
        round(results["double-turn"]["ADE"], 2), round(results["double-turn"]["FDE"], 2),
        round(results["ripple-road"]["ADE"], 2), round(results["ripple-road"]["FDE"], 2),
        round(results["all"]["ADE"], 2), round(results["all"]["FDE"], 2))
    result += "(minADE/minFDE)\t" + "\t{}/{} \t {}/{} \t {}/{} \t {}/{} \t {}/{}\n".format(
        round(results["original"]["minADE"], 2), round(results["original"]["minFDE"], 2),
        round(results["smooth-turn"]["minADE"], 2), round(results["smooth-turn"]["minFDE"], 2),
        round(results["double-turn"]["minADE"], 2), round(results["double-turn"]["minFDE"], 2),
        round(results["ripple-road"]["minADE"], 2), round(results["ripple-road"]["minFDE"], 2),
        round(results["all"]["minADE"], 2), round(results["all"]["minFDE"], 2))
    
    log_results(result, save_dir, print_result=True)


def log_results(result, path_to_images, log_result=True, print_result=False):
    if print_result:
        print(result)
    if log_result:
        string = dt.datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + "\n"
        string += "#Scenario: " + str(args.n_scenarios) + "\n"
        string += "#Trajectory: " + str(args.n_trajectory) + "\n"
        string += result
        with open(path_to_images+'/statistics.txt', 'w') as f:
            f.write(string)

def black_box_attack(args):
    """
    Use black-box optimization to find best attack powers.
    
    """
    SOR, HOR = 0, 0
    search_space = np.array([[-10, 10],
                             [-10, 10],
                             [-10, 10]])
        
    global scene_num
    for scenario in tqdm(range(args.n_scenarios)):
        augmentor.setVisualize(False)
        scene_num = scenario
        if(args.optimization_type == 'baysian'):
            result = blackboxOptimizers.bayesian_optimizer(evaluate, search_space, n_iter = 70)
            best_params = result['x']
        elif(args.optimization_type == 'evolution'):
            result = blackboxOptimizers.evolution_strategy(evaluate, search_space, n_iter = 20)
            best_params = result[0]
        elif(args.optimization_type == 'parzen'):
            best = blackboxOptimizers.parzen_estimator(parzen_transfer, search_space, n_iter=50)
            best_params = []
            for key in best.keys():
                best_params.append(best[key])
            best_params = np.array(best_params)
        else:
            raise Exception('Wrong optimization type inserted.')

        if(not args.save_addr == ""):
            augmentor.setVisualize(True)
        reward = evaluate(best_params)
        SOR += OR[0] * 100 / args.n_scenarios 
        HOR += OR[1] * 100 / args.n_scenarios 
    print("Model \t black-box attack(SOR/HOR)")
    print(args.model_name, "\t {}/{}".format(
        round(SOR), round(HOR)))

def parzen_transfer(p1, p2, p3):
    return evaluate([p1, p2, p3])

def evaluate(params):
    """
    Return a reward value based on input params
    
    Parameters
    ----------
    params : nparray
        Input attack powers for different attack functions.
    Returns
    -------
    reward : float
        A value to show how good the input parameters are. The black-box algorithms are tring to
        minimize the reward.

    """
    attack_params = {"smooth-turn": {"attack_power": params[0], "pow": 3, "border": 5},
                     "double-turn": {"attack_power": params[1], "pow": 3, "l": 10, "border": 5},
                     "ripple-road": {"attack_power": params[2], "l": 60, "border": 5}}
    save_addr = "./scenario-images/" + args.save_addr + '/' + str(scene_num) + "/0"
    OR[0], OR[1] = augmentor.attack(scene_num, attack_params, save_addr)
    lambd = 10
    reward = 0
    reward -= OR[0]
    # L0 regularization to forcing the algorithm to use one of the attack funcitions
    num = 0
    if np.abs(params[0]) > 0.5:
        num += 1
    if np.abs(params[1]) > 0.5:
        num += 1
    if np.abs(params[2]) > 0.5:
        num += 1
    if(num > 1):
        reward += (num - 1) * lambd
    return reward
    
def transfer(args):
    """
    Transfer the attack parameters obtained during attack to one of models and use them for another model.

    Parameters
    ----------
    args : 
        args.transfer from: the model which the parameters are obtained from.
        args.model_name: the model which is under the attack.

    """
    data = read('transfer-attack-files/' + args.transfer_from + '-' + str(args.n_scenarios))
    num_scenarios = len(data)
    SOR, HOR = 0, 0   
    for scenario in tqdm(range(num_scenarios)):
        save_addr = "./scenario-images", args.save_addr + '/' + str(scenario) + "/0"
        idx, attack_params, offroad = data[scenario]
        sceneSOR, sceneHOR = augmentor.attack(scenario, attack_params, save_addr)
        SOR += sceneSOR * 100 /num_scenarios
        HOR += sceneHOR * 100 /num_scenarios
    print("Transfer params from model", args.transfer_from)
    print("Model \t transferred attack(SOR/HOR)")
    print(args.model_name, "\t\t {}/{}".format(
        round(SOR), round(HOR)))

    
def read(name):
    with open(name, 'rb') as filehandle:
        h = pickle.load(filehandle)
    return h

def write(obj, name):
    with open(name, 'wb+') as filehandle:
        pickle.dump(obj, filehandle)

if __name__ == '__main__':
    main()

