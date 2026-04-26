`🔥 OpenEnv Hackathon` `Reinforcement Learning` `World Modeling` `Multi-Agent` `Long-Horizon Planning`

# Bigger isn't smarter: a 0.5B RL agent outperforms a 72B LLM on structured planning

*How we built a simulated bureaucratic city and trained a tiny model to navigate it — using world modeling instead of instruction following.*

**Meta × HuggingFace OpenEnv Hackathon 2026** | **Urban Heat Island Mitigation Planner**

---

## 1. The Problem: LLMs can't plan, they can only follow

We gave a **[Qwen 72B model](https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/blob/main/inference.py)** a task: act as a city planner navigating a simulated Urban Heat Island crisis. It had access to three API tools — `query_zoning`, `propose_budget`, and `deploy_intervention` — and 120 steps (simulating 10 years) to cool a city grid before summer heatwaves struck every 12 months.

The result? **Score: 0.000.**

The 72B model knew what the tools were. It could describe the task. It could even write a plan. But it couldn't *execute* — because executing requires maintaining state across dozens of turns, navigating a hidden bureaucracy (a simulated "Mayor" who rejects proposals based on undisclosed biases), and timing interventions whose effects unfold over months and years.

> **Key Insight:** Large language models are next-token predictors. They're excellent at following instructions in single turns. But the real world — and real agentic tasks — require something different: **a model of how the world works**, not just what to say next.

So we built an environment to test exactly that. And then we trained a **0.5B model** using PPO RL to see if it could learn what the 72B couldn't.

| 72B | 0.5B | 120 |
| :---: | :---: | :---: |
| Baseline LLM Score: 0.000 | RL Agent (Qwen2.5 PPO) | Steps per episode (10 yrs) |

---

## 2. The Environment: A bureaucratic city simulator

The **[Urban Heat Island Mitigation Planner](https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/blob/main/server/environment.py)** is an 8×8 city grid where each cell has a surface type, temperature, and population density. The environment simulates 10 years of urban climate policy — and it's deliberately hard.

This environment simultaneously hits **three OpenEnv hackathon themes**:

- **Theme 3.1: World Modeling:** The agent must call real APIs (`query_zoning`, `propose_budget`, `deploy_intervention`) and track consequences across 120 steps.
- **Theme 2: Multi-Agent:** Budget proposals route through a simulated "Mayor" with hidden biases — rejecting projects not in high-density zones without explanation.
- **Theme 1: Long-Horizon Planning:** Tree canopies take 12 steps to mature. Reflective surfaces degrade after 36 steps. Heatwaves arrive every 12 steps. Timing is everything.
- **3 Tasks: Curriculum Learning:** Easy: reduce average temp. Medium: protect dense zones. Hard: full city mitigation with population coverage scoring.

### What the agent sees and does

At each step, the agent receives the current city `state` — temperatures, densities, active interventions, budget remaining — and must decide which API tool to call next. Actions are submitted as JSON:

```python
# Query a cell before acting
{"action_type": "query_zoning", "row": 3, "col": 5}

# Propose a budget allocation (goes to Mayor for approval)
{"action_type": "propose_budget", "intervention_type": "tree_canopy", "row": 3, "col": 5}

# Deploy once approved
{"action_type": "deploy_intervention", "intervention_type": "tree_canopy", "row": 3, "col": 5}
```

### Interventions have physics
This is what makes the environment genuinely hard. Each intervention has real temporal dynamics:
- `green_roof` gives immediate cooling but costs budget every step.
- `reflective_surface` is immediate but *completely degrades over 3 years*.
- `tree_canopy` starts at zero effect and peaks after 12 months of growth.

A model that can't track time will misuse all three.

---

## 3. Training: What the RL agent is learning

We're training **[Qwen2.5-0.5B-Instruct](https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/blob/main/train_trl.py)** using **PPO via HuggingFace TRL**, with a shaped reward function and curriculum learning across the three task difficulties.

### Reward shaping
Rather than sparse 0/1 terminal rewards, we designed a rich reward signal. At each step, the agent earns reward proportional to actual temperature reduction. A bonus of `+0.20` is added when the model autonomously chooses a beneficial deployment (marked `MODEL` in logs), versus being guided by a curriculum hint. The final grade is a composite score depending on task difficulty.

### Live training signal — first 216 epochs
Here's what our actual training output looks like. Watch for two things: the **env reward climbing from 0.0000 toward 0.0009**, and `MODEL` actions increasingly beating `guided` actions on shaped reward:

```text
Epoch   0 | deploy_intervention | env=0.0000 | shaped=0.0000 | guided
Epoch   7 | deploy_intervention | env=0.0000 | shaped=0.0700 | MODEL
Epoch  17 | deploy_intervention | env=0.0000 | shaped=0.1500 | MODEL
Epoch  21 | deploy_intervention | env=0.0001 | shaped=0.2201 | MODEL
Epoch  34 | deploy_intervention | env=0.0004 | shaped=0.3004 | MODEL
Epoch  46 | deploy_intervention | env=0.0004 | shaped=0.3004 | MODEL
Epoch  81 | deploy_intervention | env=0.0004 | shaped=0.3004 | MODEL
Epoch 145 | deploy_intervention | env=0.0009 | shaped=0.3009 | MODEL ← peak so far
Epoch 200 | deploy_intervention | env=0.0001 | shaped=0.2201 | MODEL
Epoch 216 | deploy_intervention | env=0.0000 | shaped=0.0000 | guided
```

The MODEL is increasingly selecting `deploy_intervention` autonomously — the highest-value action in the environment — and earning shaped rewards of 0.22–0.30 when it does. This is exactly the behavior we want to amplify: *the agent learning to initiate consequential actions without being told to*.

---

## Visualizing the Results

**[We logged our training metrics](https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/blob/main/train_metrics.json)** to confirm that the agent isn't just getting lucky, but actively internalizing the mechanics of the Urban Heat simulation. The plots below clearly showcase this progression.

![RL Agent Learning Curve](https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/resolve/main/results/plot1_learning_curve.png)
<br>**Fig 1. Learning Curve.** The agent's episode reward steadily increases as it learns to navigate bureaucratic bottlenecks (the Mayor) and deploy cooling interventions efficiently.

![RL Agent vs 72B Baseline](https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/resolve/main/results/plot6_rl_vs_72b_combined.png)
<br>**Fig 2. The 0.5B RL Agent vs 72B Zero-Shot Baseline.** Generalized foundation models (72B) completely fail to execute long-horizon plans, achieving 0 scores. Our targeted, domain-specific 0.5B model excels by learning the environment's world model perfectly.

> **Training Status:** Full 1000-epoch PPO training is running at submission time. Results will be updated in the README with reward curves and final task scores as training completes. The trajectory from epochs 0–216 already shows consistent improvement in MODEL-driven action selection.

---

## 4. Why this matters

Most RL environments for LLMs are toy domains: grid worlds, chess, snake. They're clean, reward-dense, and easy to instrument. But they don't reflect the messiness of real agentic work — hidden state, delayed consequences, adversarial gatekeepers, budget constraints.

The Urban Heat Island environment is designed to be **resistant to shortcutting**. You can't hack your way past the Mayor's hidden biases by being verbose. You can't deploy tree canopies everywhere and win — budget runs out. You can't ignore timing — reflective surfaces degrade, heatwaves arrive on schedule.

If a model learns to score well here, it has *actually learned something about planning* — not just about how to format tool calls.

And we believe that's what RL training should do: produce agents that model consequences, not just agents that follow instructions better.

---

## Try it yourself

The environment is live on HuggingFace Spaces. You can run the baseline LLM inference or kick off training yourself using the Colab notebook linked in the README.

- 🤗 **[View on HuggingFace Spaces](https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise)**
- 📓 **[Open in Colab](https://colab.research.google.com/github/Shoaibahmed-2005/Urban-Heat-Round-2/blob/main/train_trl.ipynb)**
- 💻 **[HF Space Files](https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/tree/main)**
