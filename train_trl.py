"""
Hackathon Round 2 — TRL Training Script (FIXED VERSION)
Fixes applied:
  1. Model output is used when valid — PPO trains on actual model decisions
  2. Prompt includes grid state (top hot cells) so model can reason
  3. Reward shaping is task-aligned, not format-rewarding
  4. KL constraint loosened, batch size increased so weights can actually move
  5. Curriculum learning: task difficulty increases over epochs
"""

import torch
import random
import requests
import json
import re
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# ─────────────────────────────────────────────
# PPO Configuration
# FIX 4: Loosened KL (was 0.1→0.5), lower init_kl_coef (was 0.2→0.05)
#         so model weights can actually move during training.
#         Increased batch_size 1→4, mini_batch_size 1→2 for stable gradients.
# ─────────────────────────────────────────────
config = PPOConfig(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    learning_rate=5e-6,
    batch_size=4,
    mini_batch_size=2,
    gradient_accumulation_steps=2,
    target_kl=0.5,       # was 0.1 — too tight, froze model
    init_kl_coef=0.05,   # was 0.2 — was punishing any weight movement
    kl_penalty="kl",
)

TOTAL_EPOCHS = 1000

# ─────────────────────────────────────────────
# Curriculum task schedule
# FIX 5: Start with easiest task, unlock harder ones progressively.
#         Epoch 0-399   → reduce_avg_temp   (single objective, dense reward)
#         Epoch 400-699 → protect_dense_zones (introduces budget constraint)
#         Epoch 700+    → full_mitigation    (full task, model is now warmed up)
# ─────────────────────────────────────────────
def get_task_id(epoch):
    if epoch < 400:
        return "reduce_avg_temp"
    elif epoch < 700:
        return "protect_dense_zones"
    else:
        return "full_mitigation"

# ─────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────
def env_step(action_dict):
    try:
        r = requests.post("http://localhost:8000/step", json=action_dict, timeout=10)
        return r.json()
    except Exception:
        return {"done": True, "reward": 0.0, "state": {}}

def env_reset():
    try:
        return requests.post("http://localhost:8000/reset", timeout=10).json()
    except Exception:
        return {}

# ─────────────────────────────────────────────
# FIX 2: Prompt now includes top hot cells from state so model
#         can reason about WHERE to act, not just guess row/col blindly.
# ─────────────────────────────────────────────
def format_env_prompt(state, epoch):
    task_id = get_task_id(epoch)
    heatwave_in = state.get("next_heatwave_in", 0)
    budget = state.get("budget", 20)
    season = state.get("season", "Spring")

    # Extract top 3 hottest cells from state grid if available
    grid = state.get("grid", [])
    hot_cells = []
    if grid:
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                temp = cell.get("temperature", 0) if isinstance(cell, dict) else 0
                density = cell.get("population_density", 0) if isinstance(cell, dict) else 0
                hot_cells.append((r, c, temp, density))
        hot_cells.sort(key=lambda x: x[2], reverse=True)
        hot_cells = hot_cells[:3]

    hot_str = ""
    if hot_cells:
        hot_str = "Top hot zones (row,col,temp,density): " + \
                  ", ".join(f"({r},{c},{t:.1f},{d:.2f})" for r,c,t,d in hot_cells) + "\n"
    else:
        hot_str = "Grid data not available — use row 0-7, col 0-7.\n"

    task_hint = {
        "reduce_avg_temp":     "Goal: lower average grid temperature. Target the hottest cells.",
        "protect_dense_zones": "Goal: deploy cooling in high population density (>=0.4) cells.",
        "full_mitigation":     "Goal: reduce temp AND protect dense zones AND stay within budget.",
    }.get(task_id, "")

    return (
        f"Task: {task_id} | Season: {season} | Budget: {budget} | Heatwave in: {heatwave_in} steps\n"
        f"{hot_str}"
        f"{task_hint}\n"
        "Respond ONLY with valid JSON: {\"action_type\": \"deploy_intervention\", "
        "\"intervention_type\": \"reflective_surface\", \"row\": <0-7>, \"col\": <0-7>}\n"
        "action_type must be one of: query_zoning, propose_budget, deploy_intervention\n"
    )

def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

# ─────────────────────────────────────────────
# Smart Guided Exploration (fallback only)
# Used ONLY when model fails to produce valid JSON.
# ─────────────────────────────────────────────
TARGET_CELLS = [
    (0, 0), (0, 7), (2, 3), (3, 6),
    (4, 1), (5, 5), (7, 2), (7, 7),
]

def smart_action(epoch):
    step_in_seq = epoch % 3
    cell_idx    = (epoch // 3) % len(TARGET_CELLS)
    row, col    = TARGET_CELLS[cell_idx]
    sequence    = ["query_zoning", "propose_budget", "deploy_intervention"]
    return {
        "action_type":       sequence[step_in_seq],
        "intervention_type": "reflective_surface",
        "row": row,
        "col": col,
    }

VALID_ACTIONS       = ["query_zoning", "propose_budget", "deploy_intervention"]
VALID_INTERVENTIONS = ["green_roof", "reflective_surface", "tree_canopy"]

# ─────────────────────────────────────────────
# FIX 3: Task-aligned reward shaping
# Replaces flat +0.1 JSON bonus with meaningful sub-rewards.
# ─────────────────────────────────────────────
def compute_shaped_reward(env_reward, parsed_action, response_text, state, epoch):
    shaped = env_reward
    task_id = get_task_id(epoch)

    # Small bonus: model produced parseable JSON at all
    if re.search(r'\{[^{}]*\}', response_text):
        shaped += 0.02

    if parsed_action is None:
        return shaped

    row = parsed_action.get("row", -1)
    col = parsed_action.get("col", -1)
    action_type = parsed_action.get("action_type", "")

    # Bonus: targeting a deploy action (not just querying)
    if action_type == "deploy_intervention":
        shaped += 0.05

    # Bonus: targeting a hot cell (top of grid)
    grid = state.get("grid", [])
    if grid and 0 <= row < len(grid) and 0 <= col < len(grid[0]):
        cell = grid[row][col]
        if isinstance(cell, dict):
            temp    = cell.get("temperature", 0)
            density = cell.get("population_density", 0)

            # Bonus for targeting high-temp cell
            if temp > 36.0:
                shaped += 0.08
            elif temp > 34.0:
                shaped += 0.04

            # Bonus for targeting high-density cell (needed for Mayor approval)
            if task_id in ("protect_dense_zones", "full_mitigation") and density >= 0.4:
                shaped += 0.08

    # Bonus: env actually gave positive reward (task making progress)
    if env_reward > 0.0:
        shaped += 0.15

    return round(shaped, 4)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("Loading Model and Tokenizer...")
    from peft import LoraConfig
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name, peft_config=lora_config
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)

    print("Resetting Urban Heat Environment...")
    state = env_reset()

    print(f"Starting PPO Training Loop ({TOTAL_EPOCHS} epochs)...\n")
    epoch_rewards  = []
    shaped_rewards = []   # track shaped separately for charting
    model_used     = []   # track when model vs fallback was used

    # ─── Accumulation buffers for batch_size=4 ───────────────────────────
    batch_queries   = []
    batch_responses = []
    batch_rewards   = []

    for epoch in range(TOTAL_EPOCHS):
        task_id = get_task_id(epoch)

        # Build prompt & tokenize
        query_text   = format_env_prompt(state, epoch)
        query_tensor = tokenizer(query_text, return_tensors="pt").input_ids[0]

        # Generate from model
        response_tensor = trainer.generate(
            [query_tensor], return_prompt=False, max_new_tokens=80
        )[0]
        response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)

        # ─── FIX 1: Try to use model output first ─────────────────────────
        # Old code: ignored model output 90% of the time.
        # New code: use model output whenever it's valid. Fallback only when broken.
        parsed_action = None
        used_model    = False
        try:
            match = re.search(r'\{[^{}]*\}', response_text)
            if match:
                candidate = json.loads(match.group(0))
                if candidate.get("action_type") in VALID_ACTIONS:
                    parsed_action = candidate
                    used_model    = True
        except Exception:
            pass

        if parsed_action is None:
            # Fallback: guided action (but we still do PPO update so model learns)
            parsed_action = smart_action(epoch)
            used_model    = False

        model_used.append(1 if used_model else 0)

        # Build validated action
        action_data = {
            "task_id":           task_id,
            "action_type":       str(parsed_action.get("action_type", "query_zoning")),
            "row":               max(0, min(7, safe_int(parsed_action.get("row", 0)))),
            "col":               max(0, min(7, safe_int(parsed_action.get("col", 0)))),
            "intervention_type": str(parsed_action.get("intervention_type", "reflective_surface")),
        }
        if action_data["intervention_type"] not in VALID_INTERVENTIONS:
            action_data["intervention_type"] = "reflective_surface"

        # Environment step
        obs        = env_step(action_data)
        env_reward = float(obs.get("reward", 0.0))

        # FIX 3: shaped reward
        reward = compute_shaped_reward(env_reward, parsed_action if used_model else None,
                                       response_text, state, epoch)

        epoch_rewards.append(env_reward)
        shaped_rewards.append(reward)

        print(
            f"Epoch {epoch:4d} | task={task_id[:12]:12s} | "
            f"{action_data['action_type']:22s} | "
            f"env={env_reward:.4f} | shaped={reward:.4f} | "
            f"{'MODEL' if used_model else 'guided'}"
        )

        # Accumulate for batch update
        batch_queries.append(query_tensor)
        batch_responses.append(response_tensor)
        batch_rewards.append(torch.tensor(reward))

        # FIX 4: PPO update every batch_size steps (was every 1 step = too noisy)
        if len(batch_queries) >= config.batch_size:
            trainer.step(batch_queries, batch_responses, batch_rewards)
            batch_queries   = []
            batch_responses = []
            batch_rewards   = []

        if obs.get("done", False):
            state = env_reset()
        else:
            state = obs.get("state", state)

    # Flush remaining batch
    if batch_queries:
        trainer.step(batch_queries, batch_responses, batch_rewards)

    # ─── Save metrics + task scores ──────────────────────────────────────
    print("\nEvaluating all tasks...")
    task_scores = {}
    task_ids = ["reduce_avg_temp", "protect_dense_zones", "full_mitigation"]
    for tid in task_ids:
        try:
            result = requests.get(f"http://localhost:8000/grade/{tid}", timeout=10).json()
            task_scores[tid] = round(float(result.get("score", 0.0)), 4)
        except Exception:
            task_scores[tid] = 0.0

    model_usage_rate = round(sum(model_used) / len(model_used) * 100, 1)
    print(f"\nModel used its own output: {model_usage_rate}% of epochs")

    with open("train_metrics.json", "w") as f:
        json.dump({
            "epoch_rewards":    epoch_rewards,
            "shaped_rewards":   shaped_rewards,
            "model_used":       model_used,
            "task_scores":      task_scores,
        }, f)

    print("train_metrics.json saved.")
    print("Training complete.")

if __name__ == "__main__":
    main()
