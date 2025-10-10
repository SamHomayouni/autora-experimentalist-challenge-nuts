# Nuts Experimentalist

The Nuts Experimentalist is an intelligent agent for scientific discovery within the AutoRA framework. Its purpose is to automate the process of experimental design by deciding which experiment to run next in order to learn as efficiently as possible.

It is designed as a **hybrid active learning sampler**, which means it strategically selects data points that are expected to be the most informative, rather than relying on random or exhaustive grid searches.

---
## The Core Idea: Active Learning

Imagine you are a scout sent to map a vast, unknown territory, but you only have enough supplies to visit a hundred locations. Where do you go? Visiting locations randomly might leave huge gaps in your map. Visiting them in a grid pattern might be inefficient, spending too much time on uninteresting flatlands.

An active learning agent acts like a smart scout. It uses the information from the places it has already visited to build a rough map (the "theories") and then intelligently decides where to go next to have the biggest impact on improving that map. The goal is to create the most accurate map possible with the fewest trips.



---
## How It Works: A Hybrid Strategy

The Nuts Experimentalist uses a two-stage process that balances exploring the design space with exploiting the current theoretical models.

### Stage A: Explore or Exploit?

For each new experiment, the sampler first makes a choice:
* **Explore:** Choose a new experimental condition (a value for `n_digits`) completely at random from the set of allowed values. This is essential early in the research cycle to gather a wide range of data and prevent the models from developing biases based on a narrow initial sample.
* **Exploit:** Use the current theorist models to strategically identify the single most informative experimental condition to test next.

This strategy is "annealed," meaning the probability of exploring is high at the start and gradually decreases as more data is collected. As the models are trained on more data and become more accurate, the system increasingly trusts them to make targeted, exploitative choices.

### Stage B: The Three Guiding Principles

When the sampler decides to exploit its knowledge, it treats the theorist models as a panel of experts. It evaluates every potential experiment based on three core principles to find points where the models are weakest or most likely to be improved.

#### 🧠 Uncertainty
The sampler looks for experimental conditions where the models are most unsure about their predictions. For a given `n_digits` value, if the models' predictions for accuracy have a high variance, it indicates a region of high uncertainty. An experiment in this region provides data that helps the models reduce their uncertainty and refine their parameters.

#### 🤔 Falsification
The sampler identifies conditions where the theorist models **disagree** with each other the most. If one model predicts high accuracy for `n_digits=8` and another predicts low accuracy, this is a point of high conflict. Running an experiment at this point is guaranteed to provide evidence that challenges at least one of the current theories, forcing a significant update to the system's knowledge.

#### 🗺️ Novelty
The sampler also looks for completely new experimental conditions. It identifies values of `n_digits` that are far from any previously tested values. While the models might be confident in their extrapolations to these new regions, running an experiment there, tests the models' ability to generalize and holds the potential for discovering unexpected behavior.

After choosing a guiding principle for the current cycle, the sampler identifies the best candidate locations. To avoid running redundant experiments, it uses a **farthest-first algorithm** to ensure the final selections are not only highly-rated but also diverse and spread out.

## Inputs: What the Experimentalist Considers

To make an intelligent decision about which experiment to run next, the Nuts Experimentalist considers three key pieces of information:

* **The Design Space (`conditions`):** This is the complete menu of all possible experiments that could be run.
    * **Why:** The sampler needs to know the full range of its options before it can make a choice. It uses this pool of candidates as the starting point for its evaluation.

* **The Current Models (`models`):** This is a list of the current theorist models that have been fitted to the data collected so far.
    * **Why:** The models are the "brains" behind the operation. The experimentalist uses them as an expert panel to find points of weakness. It analyzes their predictions to measure **Uncertainty** (where a model is confused) and **Falsification** (where the models disagree with each other). Without the models, the sampler could not perform these targeted, exploitative strategies.

* **The Experimental History (`reference_conditions`):** This is a record of all the experiments that have already been run.
    * **Why:** This information is essential for the **Novelty** strategy. By comparing the full design space against the experimental history, the sampler can identify which potential experiments are in new, unexplored regions. Without this history, it would have no concept of what is "novel."