# Unbiased Estimation for Total Treatment Effect Under Interference Using Aggregated Dyadic Data

Code for paper "Unbiased Estimation for Total Treatment Effect Under Interference Using Aggregated Dyadic Data".

## Description

### Files
- simulation_total_population.py: simulation for total population experiment.
- simulation_sub_population.py: simulation for sub population experiment.
- simulation_cluster_sub_population.py: simulation for two-stafe sub population experiment.
- graph_property.ipynb: show graph property.
- plots.ipynb: show simulation results.
  
### Folders
- dataset: two network topology datasets used for simulations.
- src: simulation function.
  
### Usage
To run three types of simulations:
```
python3 simulation_total_population.py
python3 simulation_sub_population.py
python3 simulation_cluster_sub_population.py
```

use plots.ipynb to show simulation results.

### Dependencies
- Networkx >= 2.8.4
- Scipy >= 1.9.3
- Numpy >=1.23.4
- Pandas >= 1.5.1
