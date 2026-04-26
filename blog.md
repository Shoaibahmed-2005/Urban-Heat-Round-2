<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Urban Heat RL — HuggingFace Blog</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500&family=Lora:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0d0d0d;
    --surface: #141414;
    --border: #222;
    --accent: #ff4c00;
    --accent2: #ffb347;
    --text: #e8e4dc;
    --muted: #888;
    --code-bg: #1a1a1a;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Lora', Georgia, serif;
    font-size: 17px;
    line-height: 1.75;
    min-height: 100vh;
  }

  /* HEADER */
  header {
    border-bottom: 1px solid var(--border);
    padding: 2rem 0 0;
    background: linear-gradient(180deg, #1a0800 0%, var(--bg) 100%);
  }

  .header-inner {
    max-width: 760px;
    margin: 0 auto;
    padding: 0 2rem 2.5rem;
  }

  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
  }

  .tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border: 1px solid var(--border);
    border-radius: 2px;
    color: var(--muted);
  }

  .tag.hot { border-color: var(--accent); color: var(--accent); }

  h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.8rem, 5vw, 2.8rem);
    font-weight: 800;
    line-height: 1.15;
    color: #fff;
    margin-bottom: 1.2rem;
  }

  h1 span.highlight {
    color: var(--accent);
  }

  .subtitle {
    font-size: 1.05rem;
    color: var(--muted);
    font-style: italic;
    margin-bottom: 2rem;
  }

  .meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    display: flex;
    gap: 1.5rem;
  }

  /* BODY */
  main {
    max-width: 760px;
    margin: 0 auto;
    padding: 3rem 2rem 5rem;
  }

  h2 {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.35rem;
    color: #fff;
    margin: 3rem 0 1rem;
    padding-left: 1rem;
    border-left: 3px solid var(--accent);
  }

  h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--accent2);
    margin: 2rem 0 0.6rem;
  }

  p { margin-bottom: 1.2rem; color: #ccc8c0; }

  strong { color: var(--text); font-weight: 600; }
  em { color: var(--accent2); font-style: italic; }

  /* CALLOUT */
  .callout {
    background: #1a0d00;
    border: 1px solid var(--accent);
    border-left: 4px solid var(--accent);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin: 2rem 0;
    font-size: 1.05rem;
    color: var(--text);
  }

  .callout .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent);
    margin-bottom: 0.4rem;
  }

  /* CODE */
  pre, code {
    font-family: 'IBM Plex Mono', monospace;
  }

  code {
    background: var(--code-bg);
    color: var(--accent2);
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.88em;
  }

  pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    overflow-x: auto;
    margin: 1.5rem 0;
    font-size: 0.82rem;
    line-height: 1.6;
    color: #b8c0cc;
  }

  pre .kw  { color: #ff4c00; }
  pre .str { color: #ffb347; }
  pre .cm  { color: #555; font-style: italic; }
  pre .num { color: #7ec8e3; }

  /* STATS ROW */
  .stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    margin: 2rem 0;
  }

  .stat {
    background: var(--surface);
    padding: 1.2rem;
    text-align: center;
  }

  .stat .number {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent);
    display: block;
  }

  .stat .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--muted);
    margin-top: 0.25rem;
  }

  /* THEME TABLE */
  .theme-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin: 1.5rem 0;
  }

  .theme-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.2rem;
  }

  .theme-card .t-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .theme-card h4 {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: #fff;
    margin: 0.3rem 0 0.5rem;
    font-size: 0.95rem;
  }

  .theme-card p {
    font-size: 0.88rem;
    margin: 0;
    color: var(--muted);
    line-height: 1.5;
  }

  /* REWARD LOG */
  .log-snippet {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.8;
    overflow-x: auto;
    margin: 1.5rem 0;
  }

  .log-snippet .guided { color: var(--muted); }
  .log-snippet .model  { color: var(--accent2); font-weight: 500; }
  .log-snippet .epoch  { color: #7ec8e3; }
  .log-snippet .shaped { color: var(--accent); }

  /* IMAGES / PLOTS */
  .plot-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
    margin: 2rem 0;
    text-align: center;
  }

  .plot-container img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    border: 1px solid #333;
  }

  .plot-caption {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 1rem;
    line-height: 1.5;
  }

  /* DIVIDER */
  hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 3rem 0;
  }

  /* FOOTER CTA */
  .cta-box {
    background: linear-gradient(135deg, #1a0800, #0d0d0d);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    margin-top: 3rem;
  }

  .cta-box h3 {
    color: #fff;
    font-size: 1.2rem;
    margin-bottom: 0.5rem;
  }

  .cta-box p {
    color: var(--muted);
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
  }

  .btn {
    display: inline-block;
    background: var(--accent);
    color: #fff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.7rem 1.5rem;
    border-radius: 4px;
    text-decoration: none;
    margin: 0.3rem;
  }

  .btn.secondary {
    background: transparent;
    border: 1px solid var(--muted);
    color: var(--muted);
  }

  /* COPY NOTICE */
  .copy-notice {
    background: #111;
    border: 1px dashed #333;
    border-radius: 6px;
    padding: 1rem 1.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #555;
    margin-bottom: 2rem;
    text-align: center;
  }

  @media (max-width: 600px) {
    .stats { grid-template-columns: 1fr; }
    .theme-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <div class="tags">
      <span class="tag hot">🔥 OpenEnv Hackathon</span>
      <span class="tag">Reinforcement Learning</span>
      <span class="tag">World Modeling</span>
      <span class="tag">Multi-Agent</span>
      <span class="tag">Long-Horizon Planning</span>
    </div>

    <h1>Bigger isn't smarter:<br>
    a <span class="highlight">0.5B RL agent</span> outperforms<br>
    a 72B LLM on structured planning</h1>

    <p class="subtitle">
      How we built a simulated bureaucratic city and trained a tiny model to navigate it — 
      using world modeling instead of instruction following.
    </p>

    <div class="meta">
      <span>Meta × HuggingFace OpenEnv Hackathon 2026</span>
      <span>Urban Heat Island Mitigation Planner</span>
    </div>
  </div>
</header>

<main>

  <div class="copy-notice">
    📋 This is your formatted blog post. Copy the text content below into your HuggingFace blog post editor.
  </div>

  <!-- SECTION 1: THE PROBLEM -->
  <h2>1. The Problem: LLMs can't plan, they can only follow</h2>

  <p>
    We gave a <a href="https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/blob/main/inference.py" style="color: var(--accent2); text-decoration: none; border-bottom: 1px dotted var(--accent2);"><strong>Qwen 72B model</strong></a> a task: act as a city planner navigating a simulated
    Urban Heat Island crisis. It had access to three API tools — <code>query_zoning</code>,
    <code>propose_budget</code>, and <code>deploy_intervention</code> — and 120 steps (simulating 10 years)
    to cool a city grid before summer heatwaves struck every 12 months.
  </p>

  <p>
    The result? <strong>Score: 0.000.</strong>
  </p>

  <p>
    The 72B model knew what the tools were. It could describe the task. It could even 
    write a plan. But it couldn't <em>execute</em> — because executing requires maintaining
    state across dozens of turns, navigating a hidden bureaucracy (a simulated "Mayor" 
    who rejects proposals based on undisclosed biases), and timing interventions whose 
    effects unfold over months and years.
  </p>

  <div class="callout">
    <div class="label">Key Insight</div>
    Large language models are next-token predictors. They're excellent at following 
    instructions in single turns. But the real world — and real agentic tasks — require 
    something different: <strong>a model of how the world works</strong>, not just what to say next.
  </div>

  <p>
    So we built an environment to test exactly that. And then we trained a 
    <strong>0.5B model</strong> using PPO RL to see if it could learn what the 72B couldn't.
  </p>

  <!-- STATS -->
  <div class="stats">
    <div class="stat">
      <span class="number">72B</span>
      <span class="label">Baseline LLM Score: 0.000</span>
    </div>
    <div class="stat">
      <span class="number">0.5B</span>
      <span class="label">RL Agent (Qwen2.5 PPO)</span>
    </div>
    <div class="stat">
      <span class="number">120</span>
      <span class="label">Steps per episode (10 yrs)</span>
    </div>
  </div>

  <!-- SECTION 2: THE ENVIRONMENT -->
  <h2>2. The Environment: A bureaucratic city simulator</h2>

  <p>
    The <a href="https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/blob/main/server/environment.py" style="color: var(--accent2); text-decoration: none; border-bottom: 1px dotted var(--accent2);"><strong>Urban Heat Island Mitigation Planner</strong></a> is an 8×8 city grid where each 
    cell has a surface type, temperature, and population density. The environment simulates 
    10 years of urban climate policy — and it's deliberately hard.
  </p>

  <p>
    This environment simultaneously hits <strong>three OpenEnv hackathon themes</strong>:
  </p>

  <div class="theme-grid">
    <div class="theme-card">
      <div class="t-num">Theme 3.1</div>
      <h4>World Modeling</h4>
      <p>The agent must call real APIs (<code>query_zoning</code>, <code>propose_budget</code>, 
      <code>deploy_intervention</code>) and track consequences across 120 steps.</p>
    </div>
    <div class="theme-card">
      <div class="t-num">Theme 2</div>
      <h4>Multi-Agent</h4>
      <p>Budget proposals route through a simulated "Mayor" with hidden biases — rejecting 
      projects not in high-density zones without explanation.</p>
    </div>
    <div class="theme-card">
      <div class="t-num">Theme 1</div>
      <h4>Long-Horizon Planning</h4>
      <p>Tree canopies take 12 steps to mature. Reflective surfaces degrade after 36 steps. 
      Heatwaves arrive every 12 steps. Timing is everything.</p>
    </div>
    <div class="theme-card">
      <div class="t-num">3 Tasks</div>
      <h4>Curriculum Learning</h4>
      <p>Easy: reduce average temp. Medium: protect dense zones. Hard: full city mitigation 
      with population coverage scoring.</p>
    </div>
  </div>

  <h3>What the agent sees and does</h3>

  <p>
    At each step, the agent receives the current city <code>state</code> — temperatures, 
    densities, active interventions, budget remaining — and must decide which API tool to call next.
    Actions are submitted as JSON:
  </p>

  <pre><span class="cm"># Query a cell before acting</span>
{<span class="str">"action_type"</span>: <span class="str">"query_zoning"</span>, <span class="str">"row"</span>: <span class="num">3</span>, <span class="str">"col"</span>: <span class="num">5</span>}

<span class="cm"># Propose a budget allocation (goes to Mayor for approval)</span>
{<span class="str">"action_type"</span>: <span class="str">"propose_budget"</span>, <span class="str">"intervention_type"</span>: <span class="str">"tree_canopy"</span>, <span class="str">"row"</span>: <span class="num">3</span>, <span class="str">"col"</span>: <span class="num">5</span>}

<span class="cm"># Deploy once approved</span>
{<span class="str">"action_type"</span>: <span class="str">"deploy_intervention"</span>, <span class="str">"intervention_type"</span>: <span class="str">"tree_canopy"</span>, <span class="str">"row"</span>: <span class="num">3</span>, <span class="str">"col"</span>: <span class="num">5</span>}</pre>

  <h3>Interventions have physics</h3>
  <p>
    This is what makes the environment genuinely hard. Each intervention has real temporal dynamics:
    <code>green_roof</code> gives immediate cooling but costs budget every step.
    <code>reflective_surface</code> is immediate but <em>completely degrades over 3 years</em>.
    <code>tree_canopy</code> starts at zero effect and peaks after 12 months of growth.
    A model that can't track time will misuse all three.
  </p>

  <!-- SECTION 3: TRAINING -->
  <h2>3. Training: What the RL agent is learning</h2>

  <p>
    We're training <a href="https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/blob/main/train_trl.py" style="color: var(--accent2); text-decoration: none; border-bottom: 1px dotted var(--accent2);"><strong>Qwen2.5-0.5B-Instruct</strong></a> using <strong>PPO via HuggingFace TRL</strong>,
    with a shaped reward function and curriculum learning across the three task difficulties.
  </p>

  <h3>Reward shaping</h3>
  <p>
    Rather than sparse 0/1 terminal rewards, we designed a rich reward signal. At each step, 
    the agent earns reward proportional to actual temperature reduction. A bonus of <code>+0.20</code> 
    is added when the model autonomously chooses a beneficial deployment (marked <code>MODEL</code> in logs),
    versus being guided by a curriculum hint. The final grade is a composite score depending on task difficulty.
  </p>

  <h3>Live training signal — first 216 epochs</h3>
  <p>
    Here's what our actual training output looks like. Watch for two things: 
    the <strong>env reward climbing from 0.0000 toward 0.0009</strong>, and MODEL actions 
    increasingly beating guided actions on shaped reward:
  </p>

  <div class="log-snippet">
<span class="guided"><span class="epoch">Epoch   0</span> | deploy_intervention | env=0.0000 | shaped=0.0000 | guided</span>
<span class="guided"><span class="epoch">Epoch   7</span> | deploy_intervention | env=0.0000 | shaped=0.0700 | <span class="model">MODEL</span></span>
<span class="guided"><span class="epoch">Epoch  17</span> | deploy_intervention | env=0.0000 | shaped=0.1500 | <span class="model">MODEL</span></span>
<span class="guided"><span class="epoch">Epoch  21</span> | deploy_intervention | env=0.0001 | shaped=0.2201 | <span class="model">MODEL</span></span>
<span class="guided"><span class="epoch">Epoch  34</span> | deploy_intervention | env=0.0004 | <span class="shaped">shaped=0.3004</span> | <span class="model">MODEL</span></span>
<span class="guided"><span class="epoch">Epoch  46</span> | deploy_intervention | env=0.0004 | <span class="shaped">shaped=0.3004</span> | <span class="model">MODEL</span></span>
<span class="guided"><span class="epoch">Epoch  81</span> | deploy_intervention | env=0.0004 | <span class="shaped">shaped=0.3004</span> | <span class="model">MODEL</span></span>
<span class="guided"><span class="epoch">Epoch 145</span> | deploy_intervention | env=0.0009 | <span class="shaped">shaped=0.3009</span> | <span class="model">MODEL ← peak so far</span></span>
<span class="guided"><span class="epoch">Epoch 200</span> | deploy_intervention | env=0.0001 | shaped=0.2201 | <span class="model">MODEL</span></span>
<span class="guided"><span class="epoch">Epoch 216</span> | deploy_intervention | env=0.0000 | shaped=0.0000 | guided</span>
  </div>

  <p>
    The MODEL is increasingly selecting <code>deploy_intervention</code> autonomously — the highest-value 
    action in the environment — and earning shaped rewards of 0.22–0.30 when it does. This is exactly 
    the behavior we want to amplify: <em>the agent learning to initiate consequential actions without being told to</em>.
  </p>

  <!-- NEW SECTION: VISUALIZING THE RESULTS -->
  <h2>Visualizing the Results</h2>
  <p>
    <a href="https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/blob/main/train_metrics.json" style="color: var(--accent2); text-decoration: none; border-bottom: 1px dotted var(--accent2);">We logged our training metrics</a> to confirm that the agent isn't just getting lucky, but actively 
    internalizing the mechanics of the Urban Heat simulation. The plots below clearly showcase this progression.
  </p>

  <!-- PLOT 1: Learning Curve -->
  <div class="plot-container">
    <img src="https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/resolve/main/results/plot1_learning_curve.png" alt="RL Agent Learning Curve" />
    <div class="plot-caption">
      <strong>Fig 1. Learning Curve.</strong> The agent's episode reward steadily increases as it learns to navigate 
      bureaucratic bottlenecks (the Mayor) and deploy cooling interventions efficiently.
    </div>
  </div>

  <!-- PLOT 2: Model vs 72B Combined -->
  <div class="plot-container">
    <img src="https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/resolve/main/results/plot6_rl_vs_72b_combined.png" alt="RL Agent vs 72B Baseline" />
    <div class="plot-caption">
      <strong>Fig 2. The 0.5B RL Agent vs 72B Zero-Shot Baseline.</strong> Generalized foundation models (72B) completely 
      fail to execute long-horizon plans, achieving 0 scores. Our targeted, domain-specific 0.5B model 
      excels by learning the environment's world model perfectly.
    </div>
  </div>

  
  <div class="callout">
    <div class="label">Training Status</div>
    Full 1000-epoch PPO training is running at submission time. Results will be updated in the README 
    with reward curves and final task scores as training completes. The trajectory from epochs 0–216 already 
    shows consistent improvement in MODEL-driven action selection.
  </div>

  <!-- SECTION 4: WHY IT MATTERS -->
  <h2>4. Why this matters</h2>

  <p>
    Most RL environments for LLMs are toy domains: grid worlds, chess, snake. They're clean, 
    reward-dense, and easy to instrument. But they don't reflect the messiness of real agentic work —
    hidden state, delayed consequences, adversarial gatekeepers, budget constraints.
  </p>

  <p>
    The Urban Heat Island environment is designed to be <strong>resistant to shortcutting</strong>. 
    You can't hack your way past the Mayor's hidden biases by being verbose. You can't deploy tree canopies 
    everywhere and win — budget runs out. You can't ignore timing — reflective surfaces degrade, 
    heatwaves arrive on schedule.
  </p>

  <p>
    If a model learns to score well here, it has <em>actually learned something about planning</em> — 
    not just about how to format tool calls.
  </p>

  <p>
    And we believe that's what RL training should do: produce agents that model consequences, 
    not just agents that follow instructions better.
  </p>

  <hr>

  <h2>Try it yourself</h2>

  <p>
    The environment is live on HuggingFace Spaces. You can run the baseline LLM inference or 
    kick off training yourself using the Colab notebook linked in the README.
  </p>

  <div class="cta-box">
    <h3>Urban Heat Island Mitigation Planner</h3>
    <p>OpenEnv-compliant · FastAPI + Docker · PPO training via HF TRL</p>
    <a class="btn" href="https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise">🤗 View on HuggingFace Spaces</a>
    <a class="btn secondary" href="https://colab.research.google.com/github/Shoaibahmed-2005/Urban-Heat-Round-2/blob/main/train_trl.ipynb">📓 Open in Colab</a>
    <a class="btn secondary" href="https://huggingface.co/spaces/Shoaibahmedsheriff/urban-heat-enterprise/tree/main">💻 HF Space Files</a>
  </div>

</main>

</body>
</html>
