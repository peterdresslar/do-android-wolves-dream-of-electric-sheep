### Do Android Wolves Dream of Electric Sheep?

A Python ABM project that definitely uses `uv` and probably uses some other stuff.

<img src="https://raw.githubusercontent.com/peterdresslar/do-android-wolves-dream-of-electric-sheep/main/public/output_gpt-4o-mini.png">

## Project Checklist

Below is our rough roadmap:

| Level | Description                                                                 | Status |
|-------|-----------------------------------------------------------------------------|--------|
| 1     | Individual wolf agents, synchronous AI calls (everyone waits each step)     | [x]    |
| 2     | Individual wolf agents, asynchronous AI calls (continue sim while waiting)  | [ ]    |
| 3     | Throttled AI calls (poll less frequently than every step)                   | [ ]    |
| 4     | Real-time autonomous agents with continuous LLM interaction                 | [ ]    |
| 5     | Compare LLM-driven strategy vs. a machine-learning optimizer for `theta`     | [ ]    |


We also have a multiplexing factor, which is to consider models that work on a continuous chat basis versus an individual basis (i.e., how much information do we supply to the wolves, and *when*?) To start, we simply bundle up instructions step by step.
