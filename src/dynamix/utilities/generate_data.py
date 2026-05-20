from dysts import flows
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

systems_and_regimes = {
    "Aizawa": [{}],
    "AnishchenkoAstakhov": [{}],
    "Arneodo": [{}],
    "BelousovZhabotinsky": [{}],
    "Blasius": [{}],
    "CaTwoPlus": [{}, {"eps": 10, "beta": 0.5}],
    "LorenzBounded": [{}],
    "Colpitts": [{}],
    "Rossler": [{}, {"param_list": [0.05, 0.05, 6.0]}],
    "LuChenCheng": [{}],
    "NewtonLiepnik": [{}],
    "QiChen": [{}],
    "RayleighBenard": [{}],
    "RabinovichFabrikant": [{}],
    "RikitakeDynamo": [{}],
    "Rucklidge": [{}],
    "SaltonSea": [{}],
    "ShimizuMorioka": [{}, {"param_list": [0.55, 1.0]}],
    "SprottE": [{}],
    "SprottH": [{}],
    "SprottJ": [{}],
    "SprottJerk": [{}],
    "SprottK": [{}, {"param_list": [0.1]}],
    "SprottN": [{}],
    "ForcedBrusselator": [{}],
    "CellularNeuralNetwork": [{}],
    "Chen": [{}],
    "HastingsPowell": [{}],
    "ExcitableCell": [{}],
    "GlycolyticOscillation": [{}],
}

# Total points
N = 100000

# Trajectory dict
trajectories_dict = {}

for name, regimes in systems_and_regimes.items():
    SysClass = getattr(flows, name)
    for i, regime in enumerate(regimes):
        model = SysClass()
        for key, value in regime.items():
            setattr(model, key, value)
        traj = model.make_trajectory(N, resample=True, pts_per_period=50)
        key = SysClass.__name__ if len(regimes) == 1 else f"{SysClass.__name__}_regime{i+1}"
        trajectories_dict[key] = traj
    print(f"Generated trajectories for {name}")

trajectories = np.array(list(trajectories_dict.values())) # to numpy
trajectories = (trajectories - trajectories.mean(axis=(1), keepdims=True)) / trajectories.std(axis=(1), keepdims=True) # standardize

# Save trajectories to the data directory
np.save(os.path.join(DATA_DIR, "trajectories.npy"), np.swapaxes(trajectories, 0, 1))

trajs_from_system = 1000
context_len = 500
seq_len = 550

def generate_training_data(trajectories, trajs_from_system, context_len, seq_len):

    data = np.zeros(((seq_len - context_len)*2, trajs_from_system * trajectories.shape[0], trajectories.shape[2]))
    context = np.zeros((context_len, trajs_from_system * trajectories.shape[0], trajectories.shape[2]))

    for i in range(trajectories.shape[0]):
        for j in range(trajs_from_system):
            ind = np.random.randint(0, len(trajectories[i,:,0]) - seq_len)
            traj_temp = trajectories[i,ind:ind+seq_len,:]

            context[:, (trajs_from_system*i) + j,:] = traj_temp[:context_len,:]
            data[:, (trajs_from_system*i) + j,:] = traj_temp[-(seq_len - context_len)*2:,:]
    return data, context

data, context = generate_training_data(trajectories, trajs_from_system, context_len, seq_len)

# Save data and context to the data directory
np.save(os.path.join(DATA_DIR, "data.npy"), data)
np.save(os.path.join(DATA_DIR, "context.npy"), context)


# Generate test system
traj = flows.Lorenz().make_trajectory(13000, resample=True, pts_per_period=50)
traj = (traj - traj.mean(axis=(0), keepdims=True)) / traj.std(axis=(0), keepdims=True) # standardize
np.save(os.path.join(DATA_DIR, "test_system.npy"), traj[1000:, np.newaxis, :])