# Communication-Aware Multi-Agent Exploration

This repository implements **communication-aware exploration** using multi-agent reinforcement learning, **built upon the [large-scale-DRL-exploration
](https://github.com/marmotlab/large-scale-DRL-exploration) framework** developed by Marmot Lab. **Communication logic adapted from the [ IR2 framework (IR2-Multi-Robot-RL-Exploration) ](https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration)**,  originally developed by Marmot Lab.


The project focuses on enabling multiple agents to explore unknown environments while considering communication constraints. It supports visual debugging and flexible configuration for different exploration scenarios.

## ⚙️ Key Configurations

You can modify the exploration problem setup in [`parameter.py`](parameter.py):

* `N_AGENTS`: Number of agents
* `SENSOR_RANGE`: Agent sensing range
* `COMMS_RANGE`: Communication range between agents

> **Note:** The current parameters are set for visualization and debugging purposes. Adjust them as needed for your use case.

## 🚀 How to Run

Run the main script:

```bash
python driver.py
```

This will launch the multi-agent exploration simulation.

## 📂 Code Structure

| File                                | Purpose                                                                     |
| ----------------------------------- | --------------------------------------------------------------------------- |
| `driver.py`                         | Main script to initialize and run exploration                               |
| `env.py`                            | Environment definition; includes all communication and belief-sharing logic |
| `multi_agent_worker.py`             | Controls the agent rollout and deployment process                           |
| `agent.py`                          | Defines per-agent decision-making logic                                     |
| `node_manager.py`                   | Constructs and manages exploration nodes                                    |
| `parameter.py`                      | Stores configuration like agent count, ranges, etc.                         |
| `sensor.py`, `quads.py`, `utils.py` | Utility functions for sensing, geometry, and visualization                  |
| `model/`                            | Folder for pretrained models                                                |
| `maps_medium/`                      | Contains example exploration maps                                           |

## 📌 Notes

* Communication is modeled as range-limited information exchange.
* Designed for rapid prototyping and debugging of communication-aware behavior.
* Built on top of a well-tested exploration framework (ariadne2).

Feel free to open issues or contribute if you're interested in extending this work!
