import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser(description="test drawer")
parser.add_argument('--resultpath', type=str, required=True,
        help='path to the npz file directory')
parser.add_argument('--num-batches', type=int, required=True, default=10,
        help= 'the number of batches during the test')
parser.add_argument('--num-traj', type=int, required=True, default=20,
        help='the number of trajectories per batch')
args = parser.parse_args()

Results = np.load(os.path.dirname(os.path.realpath(__file__)) + '/' + args.resultpath + '/results.npz', allow_pickle = True)
returns = Results['train_returns']


valid_returns = Results['valid_returns']

print(returns)

mean_return = np.mean(returns, axis=1)
mean_valid_return = np.mean(valid_returns, axis=1)

results, valid_results = [], []
for i in range(args.num_batches):
    valid_results.append(np.mean(mean_valid_return[i*args.num_traj:(i+1)*args.num_traj]))
    results.append(np.mean(mean_return[i*args.num_traj:(i+1)*args.num_traj]))

valid_results, results = np.array(valid_results), np.array(results)

print(np.mean(valid_results), np.var(valid_results))



plt.plot((valid_results), 'g^')
plt.title('valid results')
plt.show()

print(np.mean(results), np.var(results))

plt.plot((results), 'r^')
plt.title('fast adaptation train results')
plt.show()     