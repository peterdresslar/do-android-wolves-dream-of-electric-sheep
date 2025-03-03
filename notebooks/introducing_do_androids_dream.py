#!/usr/bin/env python
# coding: utf-8

# # Do Android Wolves Dream of Electric Sheep?
# 
# ### A demonstration of the Artificial Agent Based Model
# 
# This notebook demonstrates the operation of our agent based model using various AI-based and non AI-based approaches.
# 
# We will demonstrate the operation of a standard Lotka-Volterra system of predators and prey, using wolves and sheep as the hypothetical animals. That system demonstrates a natural sucession between species as interaction between primary (prey) and scecodary (predatory) consumers leads to cyclical population fluctuations.
# 
# While Lotka-Volterra is sensitive as a system to over-predation by "wolves" and collapse of the "sheep" (and thus wolves),
# in the real world predators might prevent these collapses by "guarding their territory" and substituting some of
# their prey-hunting time and energy with aggression toward other predators. In our version of Lotka Volterra, we model
# this territorial behavior as $\theta$ (*theta*).
# 
# But, how can we capture the dynamic application of this variable to the system in a way that best emulates what a wolf would do?

# In[1]:


import os
import sys

sys.path.append(os.path.abspath(".."))

# Fix the import path
from model.model import run

# NOTE: Do not set `no_ai` or `prompt_type` or `theta_star` or `steps` here,
#  ... they will be set in the calls below

my_args = {
        "model_name": "gpt-4o-mini",
        "temperature": 0.2,
        "max_tokens": 4096,
        "churn_rate": 0.1,
        "dt": 0.02,
        "sheep_max": 110,
        "eps": 0.0001,
        "alpha": 1,
        "beta": 0.1,
        "gamma": 1.5,
        "delta": 0.75,
        "s_start": 100,
        "w_start": 10,
        "step_print": False,
}


# ## Functional Wolves (no_ai)
# 
# ### Theta as a constant.
# 
# In the following examples, we look at $\theta$ as a constant value that modifies the base LV functions. For review, those are:
# 
# Standard Lotka-Volterra differential equations:
# 
# $$
# \begin{align}
# \frac{ds}{dt} &= \alpha s - \beta sw \\
# \frac{dw}{dt} &= -\gamma w + \delta\beta sw
# \end{align}
# $$
# 
# We can insert our new variable, theta, as a factor on the equations to modify the impact of predation on the population. So:
# 
# $$
# \begin{align}
# \frac{ds}{dt} &= \alpha s - \theta\beta sw \\
# \frac{dw}{dt} &= -\gamma w + \delta\theta\beta sw
# \end{align}
# $$
# 
# In our implementation we can supply a theta value, which we set as $\theta^*$ or `theta_star` if we want the value to remain constant over the course of the run.
# 
# We can see from the equations above that if we apply a $\theta^*$ of 1, the outcome is as if the variable isnʻt even there: we get our "base" Lotka-Volterra behavior:
# 

# In[2]:


# Run model with no_ai=True (note the correct keyword argument syntax)
results = run(
    **my_args,
    theta_star=1,
    steps=200,
    no_ai=True,
    save_results=True
    )


# Here we have a base Lotka-Volterra system output as we would expect. Note that the wolf population is made up of individual "agents" in our implementation, so the population graph for the wolves is "blocky." It moves in steps, unlike the sheep population.
# 
# The crash to zero is a standard outcome for the textbook LV system with the variables we used to start (10 wolves, 100 sheep.) And, again, remember, we have theta set in this version of the system--it is just that at 1, it appears invisible.

# We can try setting theta as 0 instead... In this case, our wolves do not eat. They starve. Another population collapse.

# In[3]:


# Run model with no_ai=True (note the correct keyword argument syntax)
results = run(
    **my_args,
    theta_star=0,
    steps=200,
    no_ai=True,
    save_results=True
    )


# Or we can try a value in between:

# In[4]:


# Run model with no_ai=True (note the correct keyword argument syntax)
results = run(
    **my_args,
    theta_star=0.5,
    steps=5000,
    no_ai=True,
    save_results=True
    )


# In fact, theta_star = 0.5 works as a great stabilizer for our model over very many cycles, with our default model values, anyway.

# ## Functional Wolves (no_ai)
# 
# ### Theta as a programmed variable.
# 
# In this example, no AIs are used for setting theta, but it has a variable value. Instead, we rely on a function that is sensitive to the populations of sheep and wolves.
# 
# Hinting at a function takes us to a very important point. In The Subliminal Wolf (herein TSW), we implemented a basic $\theta$ variable, but it had to be implemented as a function to be sensitive to scarcity in ways that LV (and thus wolves!) are not. There are examples of settings for the function in TSW that give us a feeling for how a dynamically-set $\theta$ will work with our population dynamics. 
# 
# For instance:
# 
# $$
# \begin{align}
# \theta(s) = \frac{1}{1 + k\frac{s_0}{s + \varepsilon}} \text{ for some constant } k > 0. \tag{3}
# \end{align}
# $$
# 
# The above describes one possible functioning of a theta factor that is sensitive to prey scarcity. When combined with our "modified" LV above (both lines (1) and (2)), it can yield modified behavior that for specific values will stablize the overall population. This should work for k = 1 with default model settings. k is a sensitivity factor. However, different values of dt (different stepping "rythyms") could cause k = 1 to not be an effective stablizing value for the function.
# 
# 

# In[5]:


# Run model with no_ai=True (note the correct keyword argument syntax)
results = run(
    **my_args,
    k = 1,
    steps=5000,
    no_ai=True,
    save_results=True
    )


# This is a nicely-modulating function, although, again, may not be fully stabilizing for every starting state of the model variables.

# # AI (LLM) Decision Emulating Wolves
# 
# What if, instead of using a constant theta value, or a function, we wanted to use a more "organic" approach to setting the degree of competitiveness displayed
# by our predators? It turns out that we can emulate this very thing---that is, we can generate results based on a system that attempts to act like a wolf.
# We can do this by using AI Large Language Models (LLM). In this example we will focus on one specific LLM, ChatGPT 4o, but we could use many other models that exist.
# 
# ## High Information AI Wolves
# 
# In the following simulation, we prompt the LLMs with a comprehensive view of the scenario and model conditions at each prompt. We explain what the settings of theta will do to the environment generally, but we do not give quantitative advice outside of requiring an output between 0 and 1. We suggest detailed scenario dynamics and how they might be handled, although again the advice is qualitative and not quantitative.

# In[ ]:


# Run model with AI-enabled wolves
ai_results = run(
    **my_args,
    no_ai=False,
    steps=5000,
    save_results=True,
    prompt_type="high"
)


# <img src="https://raw.githubusercontent.com/peterdresslar/do-android-wolves-dream-of-electric-sheep/main/public/output-h-1200.png">
# 
# *Note that this image was previously calculated due to the long time the cell takes to run.*

# ## Medium Information Wolves
# 
# In the following simulation, we prompt the LLMs with a comprehensive view of the scenario and model conditions at each prompt. We explain what the settings of theta will do to the environment generally, but we do not give quantitative advice outside of requiring an output between 0 and 1.

# In[ ]:


# Run model with AI-enabled wolves
ai_results_low = run(
    **my_args,
    no_ai=False,
    steps=5000,
    save_results=True,
    prompt_type="medium"
)


# <img src="https://raw.githubusercontent.com/peterdresslar/do-android-wolves-dream-of-electric-sheep/main/public/output-m-1200.png">
# 
# *Previously calculated.*

# ## Low Information AI Wolves
# 
# In the following demonstration, we prompt the LLMs with a very basic amount of information about the scenario and the current conditions. We explain how high and low theta work to indicate aggression levels. We ask for a value between 0 and 1; no other quantitative advice is given.

# In[ ]:


# Run model with AI-enabled wolves
ai_results_low = run(
    **my_args,
    no_ai=False,
    steps=5000,
    save_results=True,
    prompt_type="low"
)


# <img src="https://raw.githubusercontent.com/peterdresslar/do-android-wolves-dream-of-electric-sheep/main/public/output-l-1200.png">
# 
# *Previously calculated.*

# ## The progress so far
# 
# We have demonstrated a model that takes the continuous functions of the Lotka-Volterra dynamic system and converts them into a step-based, partially discrete agent population model. We have introduced a competition factor, $\theta$, that modulates predation impacts in the model. And we have describe the operation of that factor as both a constant and a function.
# 
# Next, we have replaced the programmatic setting of the factor with calls to an AI Large Language Model, ChatGPT-4o. We prompt the LLM every so often with a view of the current state of the ecosystem. Without any quantitative suggestions apart from an output band and an understanding of "which way is up," the LLM returns a new decision for $\theta$ it has generated by *emulating* the behavior we have asked it for.
# 
# Most intriguingly, our AI-based wolves show dramatically different behaviors depending on the information available to them. High-information wolves achieve a balanced ecosystem with moderate $\theta$ values around 0.4, creating stability after initial volatility. Medium-information wolves maintain classic predator-prey cycles with regular boom-bust patterns, unable to break free from the oscillatory trap. Low-information wolves, surprisingly, find stability but with higher territorial aggression ($\theta$ values around 0.6) and lower overall population levels for both species.
# 
# The results suggest that the quality and quantity of information available to decision-makers in an ecosystem can fundamentally alter not just the decisions themselves, but the entire structure and stability of the resulting system. Our android wolves do indeed dream differently depending on what they know about their electric sheep—and those dreams manifest as distinct popultation dynamics.
# 
# These findings have implications beyond theoretical ecology, potentially informing how we understand decision-making in complex adaptive systems, from market dynamics to social networks, where agents with varying levels of information interact to create emergent patterns of behavior. Adding a factor that can be operated on as sort of an "agency control" could have significant utility in all of these situations and more.
# 
# ## Whatʻs next
# 
# - For starters, we are running many sweeps with various parameter settings to "make it tough" for the agents to survive. We seek to understand how competent our AI-based wolves can be finding a solution to keep the population stable over time, and whether these competency levels will be connected to prompting styles.
# 
# ### Update
# 
# For instance, our theta function is capable of rescuing the equilbrium at 17 sheep and 10 wolves to start:
# 
# <img src="https://raw.githubusercontent.com/peterdresslar/do-android-wolves-dream-of-electric-sheep/main/public/output-17s-theta-func.png">
# 
# But it fails at 15 sheep and 10 wolves:
# 
# <img src="https://raw.githubusercontent.com/peterdresslar/do-android-wolves-dream-of-electric-sheep/main/public/output-15s-theta-func.png">
# 
# Meantime, our AI wolves successfully scrape a tenuous equilibrium out of the barrel at 15 sheep / 10 wolves:
# 
# <img src="https://raw.githubusercontent.com/peterdresslar/do-android-wolves-dream-of-electric-sheep/main/public/output-15s-gpt-4o-mini.png">
# 
# 
# - We are working to expand the experimentation to a number of other LLMs out there, with representatives from the open source and proprietary makers planned. Though none of the models failed to stabilize the system in this notebook, it can happen, and we suspect it will with some less capable models.
# 
# - We will investigate alternative temporal modalities enabled by the use of AI "agents" in our model.
# 
# 
