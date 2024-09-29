import pickle
from pprint import pprint

def inspect_pickle_file(filename):
    with open(filename, 'rb') as f:
        state = pickle.load(f)  # Load the pickled data
    return state  # Return the loaded state to inspect

# Load and pretty-print the contents of the pickle file
state = inspect_pickle_file('result_dir/state/bandit_state.pkl')
pprint(state)