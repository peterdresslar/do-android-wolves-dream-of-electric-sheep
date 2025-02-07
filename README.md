### do-android-wolves-dream-of-electric-sheep

An investigation of agent-based modeling, LLM interactions, and the representation of time.

For our investigation, we will use the Lotka-Volterra functions to operate a finite population of 
"wolves" and "sheep": both designed as classes of agents (`agents.py`) bounded by a very simple domain
(`domain.py`). Populations of wolves and sheep will follow the predator-prey dynamics of Lotka-Volterra,
with one exception: as seen in our notebook "The Subliminal Wolf," the wolves will be given an additional
"agency" factor (`theta`) that will modify their impact on the dynamic system. Thus:

```math
dS/dt = alpha * S - beta * factor(theta) * S * W
```
```math
dW/dt = -gamma * W + delta * S * W
```

Here:
- `S` = number of sheep
- `W` = number of wolves
- `alpha` and `gamma` are the usual growth/death factors
- `beta` is a predation factor
- `delta` is a growth factor for the predators

And `agent_factor(theta)` is a function of `theta` and the scarcity of sheep. It turns out that `theta` alone can prevent sheep extinction, though only for certain values. We start by initializing `theta = 1.0`, so wolves begin as “classic” Lotka-Volterra predators.

---

## Getting Started with AI Agency

In this project, we hand over the tuning of `theta` to an AI model. We give the wolves’ situation (like current counts of sheep and wolves) to an LLM, asking it to:
1. Avoid letting the wolf population reach zero.
2. Maximize the wolf population over time.

We can vary how much information the LLM sees—e.g., whether it’s aware of Lotka-Volterra theory or not. Our simplest experiment has a single LLM setting `theta` for the entire system. Eventually, we can compare these results to a machine-learning approach to see which strategy is better at keeping the wolves alive and thriving.

---

## Getting Personal

From there, we can split the wolf population into individual agents while still treating sheep as a single (continuous) resource. This setup more closely matches an agent-based model, where each wolf has its own decision-making loop.

As we give more autonomy to each agent, we face interesting **time** challenges:
- LLM calls are slow (often seconds per call) compared to our simulation steps (milliseconds).
- We must decide how often to request a “decision” from each wolf. 
- We might pause our simulation each step to wait for LLM responses, or run asynchronously and apply the LLM’s advice whenever it arrives.

---

## Time Representation & LLM Calls

For discrete time steps, we might:
- Pause the simulation every step until the LLM returns a `theta`.
- Poll the LLM less frequently than every step (e.g., every N steps).
- Poll asynchronously (keep moving until the LLM responds, then apply the response at some point).

In a continuous-time or “real-time” model, things get even trickier; we could let the AI “choose” when to act without a strict step schedule. That might eliminate the notion of discrete time steps entirely, requiring a more event-driven approach.

---

## Project Checklist

Below is our rough roadmap:

| Level | Description                                                                 | Status |
|-------|-----------------------------------------------------------------------------|--------|
| 0     | Basic LV + single AI-controlled `theta` parameter                           | [ ]    |
| 1     | Individual wolf agents, synchronous AI calls (everyone waits each step)     | [ ]    |
| 2     | Individual wolf agents, asynchronous AI calls (continue sim while waiting)  | [ ]    |
| 3     | Throttled AI calls (poll less frequently than every step)                   | [ ]    |
| 4     | Real-time autonomous agents with continuous LLM interaction                 | [ ]    |
| 5     | Compare LLM-driven strategy vs. a machine-learning optimizer for `theta`     | [ ]    |

---

## Getting Started

- **Run the Notebook**: Clone this repo and open `notebooks/agent-wolf.ipynb`.  
- **Check Out**: `agents.py` and `domain.py` for how sheep and wolves are defined.  
- **Explore**: Our “Subliminal Wolf” notebook for the idea behind the `theta` parameter and `agent_factor`.

We hope this project provides a fun demonstration of both agent-based modeling and integrating AI into a predator-prey scenario. Stay tuned as we add more features and climb through the levels of complexity!
