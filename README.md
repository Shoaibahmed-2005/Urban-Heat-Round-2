---
title: Urban Heat Env (Triple Threat Edition)
emoji: 🌡️
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---

# Urban Heat Island Mitigation Planner — Round 2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shoaibahmed-2005/Urban-Heat-Round-2/blob/main/train_trl.ipynb)
[![Hugging Face Space](https://img.shields.io/badge/HF%20Space-Urban%20Heat%20Enterprise-blue)](https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise)

An intricately designed, multi-theme RL environment built on top of the **OpenEnv** framework for the Meta × HuggingFace OpenEnv Hackathon.

## 🚀 Hackathon Quick Links
* **Hugging Face Space**: [Shoaibahmedsheriff/urban-heat-enterprise](https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise)
* **Training Script (Colab)**: [train_trl.ipynb](https://colab.research.google.com/github/Shoaibahmed-2005/Urban-Heat-Round-2/blob/main/train_trl.ipynb)
* **Project Demo (Blog Format)**: [blog.md](blog.md) (Structured to follow the official 5-step demo rubric)
* **Architecture Deep Dive**: See `project_handoff.md` for our full Hackathon strategy.

## 🌍 Motivation

Urban Heat Islands (UHIs) are metropolitan areas that are significantly warmer than their surrounding rural areas due to human activities. This excess heat leads to increased energy consumption, elevated emissions of air pollutants, and compromised human health. 

Our environment simulates the complexities of mitigating UHIs at a city-planning scale. Planners cannot simply "place" cooling interventions everywhere. They must manage a constrained budget, navigate bureaucratic zoning policies, and account for long-term delays (e.g., waiting a year for a tree canopy to grow). By modeling this as a Reinforcement Learning problem, we can train AI agents to formulate optimal, multi-year strategic plans to cool down high-density populations efficiently.

## ⚙️ How the Environment Works

This environment perfectly encapsulates three core Hackathon themes:
1. **World Modeling (Enterprise Workflows):** The agent must navigate simulated APIs (`query_zoning`, `propose_budget`, `deploy_intervention`) to get anything built. Built natively on **OpenEnv**.
2. **Multi-Agent Interactions:** Budget proposals are routed through a simulated "Mayor" actor who possesses hidden biases (e.g., rejecting projects not located in high-density areas).
3. **Long-Horizon Planning:** The simulation runs for 120 steps representing 10 years. Interventions like `tree_canopy` take 12 months to grow, while `reflective_surface` degrades over 3 years. The agent must successfully prepare for "Summer Heatwaves" that spawn every 12 months.

### State & Action Space

**Action Space (API Tool Calling)**
Actions are submitted as JSON requests to the environment:
- **`query_zoning`**: `{"action_type": "query_zoning", "row": <0-7>, "col": <0-7>}`
- **`propose_budget`**: `{"action_type": "propose_budget", "intervention_type": "<type>", "row": <0-7>, "col": <0-7>}`
- **`deploy_intervention`**: `{"action_type": "deploy_intervention", "intervention_type": "<type>", "row": <0-7>, "col": <0-7>}`

**State & Observation**
The State consists of an 8x8 city grid where each cell has properties like surface type, temperature, and population density. The environment also tracks global state such as budget, step count, active interventions, and proposals.

### Interventions & Dynamics
- `green_roof`: High immediate cooling, charges a 0.1 budget maintenance fee/step.
- `reflective_surface`: Immediate cooling, completely degrades over 36 steps (3 years).
- `tree_canopy`: Peak cooling, but starts at 0 and takes 12 steps (1 year) to fully mature.

#### Anti-Reward Hacking
To prevent the RL agent from spamming actions to farm points, the environment logic explicitly ties rewards to genuine state changes and enforces budget constraints:
```python
if action == "deploy_intervention":
    if self.budget < intervention.cost:
        return PENALTY_NO_BUDGET # Prevents action spam
    if intervention == "tree_canopy":
        cell.cooling = 0.0       # Forces long-horizon planning
        self.growth_queue.append({"mature_in": 12})
```

---

## 📈 Training Evidence & Results

We successfully trained an agent using Hugging Face TRL (PPO framework) to solve this complex bureaucratic and environmental puzzle. The training script (`train_trl.py` / `train_trl.ipynb`) connects to the FastAPI backend and trains an LLM to issue valid tool-call sequences.

**Training Logs:** We have included the raw step-by-step metrics and reward outputs from our run in `train_metrics.json` for full transparency and reproducibility.

### Phase 1: The Failure of Foundation Models (72B)
Without a true world model, even massive 72B parameter models fail at this task. Because the agent must wait for budget approvals and factor in delayed physics, zero-shot models get trapped in repetitive loops.

![72B Hallucination Loop](results/plot5_72b_loop_proof.png)
*(Above: The "Loop of Death". The 72B model repeatedly queries the exact same API without ever advancing the environment state.)*

### Phase 2: Training the 0.5B RL Agent
We trained a tiny 0.5B model specifically to internalize the environment's dynamics. Over the training epochs, we can see the model transition from random exploration to consistent, high-reward behavior.

![Training Progress Windows](results/plot4_progress_windows.png)
*(Above: Breaking the training into windows shows the agent moving from low-reward exploration early on to consistent task completion.)*

![Model Autonomy vs Curriculum Guidance](results/plot2_model_vs_guided.png)
*(Above: Transitioning to Autonomy. As training progresses, autonomous "MODEL" actions completely overtake curriculum-guided actions.)*

![Learning Curve](results/plot1_learning_curve.png)
*(Above: The final learning curve showing the agent's episode reward steadily increasing as it navigates the Mayor's bottlenecks.)*

### Phase 3: The Final Verdict
By prioritizing a world model over sheer parameter count, our targeted 0.5B RL agent was able to completely dominate the generalized foundation model.

![Performance Comparison](results/plot6_rl_vs_72b_combined.png)
*(Above: The 0.5B RL agent vs the 72B Zero-Shot Baseline. The ultimate proof that for structured planning, bigger isn't always smarter.)*

---

## 🛠️ Building and Using Your Environment

This environment is fully compatible with the **OpenEnv standard** (see `openenv.yaml` for tasks and API routing).

### Setup and Local Execution

```bash
# Install dependencies (using pip or uv)
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
```

### Running the Server & Dashboard

```bash
# Start the simulation backend
uvicorn server.app:app --host 0.0.0.0 --port 8000

# To view the visual dashboard, simply open dashboard.html in your browser!
```

### Docker (Hugging Face Spaces)
A `Dockerfile` is included for easy deployment. Note that the Docker image exposes port `7860`.
```bash
docker build -t urban-heat-env .
docker run -p 7860:7860 urban-heat-env
```

## 🧠 Running RL Training & Inference

To verify our ability to solve this complex space, you can run the provided Hugging Face TRL PPO training script:

```bash
python train_trl.py
```
*(Note: Requires valid Hugging Face authentication if running outside of Colab)*

We also provide a baseline LLM inference script to interact with the environment using models like Qwen:

```bash
python inference.py
```

## 📁 Project Structure

```text
urban_heat_env/
├── README.md             # Environment documentation
├── blog.md               # Formatted Hugging Face blog post
├── models.py             # Action, Observation, State definitions
├── inference.py          # Baseline LLM inference script
├── train_trl.py          # RL Training using Hugging Face TRL
├── train_metrics.json    # Raw RL training metrics and reward logs
├── dashboard.html        # Interactive visual dashboard
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image definition
├── openenv.yaml          # OpenEnv configuration
├── results/              # Training plots and media
├── scripts/              # Utility scripts (plotters, updaters)
└── server/
    ├── __init__.py
    ├── environment.py    # Core CityGrid environment logic
    └── app.py            # FastAPI application
```