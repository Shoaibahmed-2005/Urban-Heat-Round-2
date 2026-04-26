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
* **Demo Video**: [Insert YouTube Video Link Here] *(Judge note: Replace with actual URL)*
* **Writeup/Blog Post**: [blog.md](blog.md) (Formatted for Hugging Face)
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

---

## 📈 Training Evidence & Results

We successfully trained an agent using Hugging Face TRL (PPO framework) to solve this complex bureaucratic and environmental puzzle. The training script (`train_trl.py` / `train_trl.ipynb`) connects to the FastAPI backend and trains an LLM to issue valid tool-call sequences.

**Training Logs:** We have included the raw step-by-step metrics and reward outputs from our run in `train_metrics.json` for full transparency and reproducibility.

### Learning Curve
The agent progressively learned to navigate the API, getting its budget proposals approved and deploying interventions to reduce the city's temperature.

![Learning Curve](results/plot1_learning_curve.png)
*(Above: Average episode reward improving over training steps as the agent learns the environment dynamics.)*

### Model vs Baseline
Our trained RL agent (0.5B parameters) successfully learned to outperform significantly larger foundation models (72B) by properly sequencing API calls and planning for long-horizon delays.

![Performance Comparison](results/plot6_rl_vs_72b_combined.png)
*(Above: A comparison demonstrating the superiority of our targeted RL training approach versus zero-shot generalized models.)*

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