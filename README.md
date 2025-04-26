# Traffic Signal Control with Shared-Policy DQN and CTDE under Emergency and Fault Scenarios

## Overview
This project implements an adaptive traffic light control system using Deep Q-Network (DQN) with a shated policy across intersections.

The agent learns dynamic phase switching policies to minimise traffic delay, clear emergency vehicles faster, and adapt to signal faults.

The system supports centralized training with decentralized execution (CTDE) and is built on top of the SUMO traffic simulator, integrated using the TraCI API.

## Project Structure
```
TLCS/
├── intersection/                     
├── logs/                              
├── logs2/                            
├── logs25/                            
├── models/
│   └── model_339/shared_policy/       
├── rl_models/                         
├── communication.py                   
├── emergency_handler.py              
├── generator.py                      
├── intersection_config.py             
├── memory.py                          
├── model.py                           
├── testing_main.py                   
├── testing_settings.ini               
├── testing_simulation.py              
├── training_main.py                   
├── training_settings.ini              
├── training_simulation.py             
├── utils.py                             
└── README.md                          
```

## Requirements
- Python 3.9+

- TensorFlow 2.8+

- SUMO Simulator 1.12+ (must be installed separately)

- TraCI4SUMO Python API

- Other Python packages:

    - numpy

    - pandas

    - matplotlib

### Environment Setup
**1. Create and activate a virtual environment.**
```
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```
\
**2.Install dependencies using:**
```
pip install -r requirements.txt
```

## How To Run
### 1. Model Training
Navigate to the project root ```TLCS``` folder:
``` 
cd TLCS 
```
\
Train the DQN-based traffic control agent:
```
python training_main.py
```
- The agent will interact with the environment, collect experience and update the Q-network.

- Model will be saved in the ```models/``` directory

- Training logs (rewards, delay, queue) will be saved in the ```logs/``` directory.


### 2. Testing (Evaluation)
To evaluate the trained model:

```
python testing_main.py
```

- Evaluates across different intersection types and random seeds.

- Saves evaluation metrics (average delay, queue length, throughput) in CSV files.

## Core Components
*Lane embedding network*
- A two-layer MLP that processes per-lane features independently.


*Mean Pooling Aggregator*
- Aggregates all lane embeddings into a single vector to maintain permutation-invariance.


*Q-Network*
- 	Fully connected layers that output Q-values for each traffic light phase


*Replay Memory*
- 	Stores past experiences (state, action, reward, next_state, done) for experience replay.


*Epsilon-Greedy Policy*
- Balances exploration and exploitation during training.


*Centralized Critic*
- During training, a centralized critic may be used for CTDE (centralized training, decentralized execution).


## Intersection Types
- Cross Intersection `cross`
- Roundabout `roundabout`
- 1x2 Grid `1x2_grid`
- Double-T intersection `double_t`
- T-Intersection with U-Turn `t_with_u_turn`

## Metrics Recorded
During both training and testing:
- Delay per vehicle
- Average queue length
- Throughput
- Cumulative negative rewards
- Total emergency vehicle delay
- Average emergency vehicle delay
- Emergency delay ratio


## Additional Features
**Emergency Vehicles Handling**
- Emergency vehicles are prioritised by dynamically adjusting the active green phases.

**Faulty Signal Handling**
- Random signal faults (green-to-red failures) are injected; agent must learn to adapt

**Support for Multiple Intersection Types**
- Easily add new intersections (e.g., roundabout, T-junctions) via ```intersection_config.py```.

**Shared Policy across Intersections**
- A single policy model is trained to generalise across multiple intersection types.
