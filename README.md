# Few-Shot Learning for Spam Detection with Large Language Models

## Introduction
In this project, you will explore various techniques to enhance the performance of large language models (LLMs) on a spam detection task. 

Our target is to utilize a pre-trained LLM and apply different strategies, including zero-shot learning, naive prompting, and full fine-tuning, to classify emails as spam or not spam. The main examples we provide are based on Bayesian inverse classification methods. But you are encouraged to explore any methods you are interested in.

### Examples Provided
1. **Chatbot Example**: 

    `examples/chatbot_example.py` demonstrates general text generation; 
    
    run it with `uv run -m examples.chatbot_example` and experiment with several prompts.

2. **Bayes Inverse Zero Shot Example**: 

    `examples/bayes_inverse_zero_shot.sh` launches `uv run -m examples.bayes_inverse --method zero_shot` to evaluate the zero-shot baseline without extra training.
3. **Bayes Inverse Naive Prompting Example**: 

    `examples/bayes_inverse_naive_prompting.sh` wraps `uv run -m examples.bayes_inverse --method naive_prompting` and injects a richer prompt at inference time.
4. **Bayes Inverse Full Finetune Example**: 

    `examples/bayes_inverse_full_finetune.sh` triggers `uv run -m examples.bayes_inverse --method full_finetune` to fine-tune the model before evaluation.

### Bayesian Inverse Classification
The Bayesian inverse classification method leverages the generative capabilities of LLMs to perform classification tasks. By modeling the joint distribution of inputs and labels, we can compute the posterior probabilities of labels given the inputs, allowing for effective classification.

$$
\begin{aligned}
P_\theta(Y_\text{label}|X_{\leq i}) &= \frac{P_{\theta}(X_{\leq i}|Y_\text{label})P_{\theta}(Y_\text{label})}{P_{\theta}(X_{\leq i})}\\
&= \frac{P_{\theta}(X_{\leq i}|Y_\text{label})P_{\theta}(Y_\text{label})}{\displaystyle\sum_{Y'}P_{\theta}(X_{\leq i}|Y')P_{\theta}(Y')}\\
&= \frac{P_{\theta}(X_{\leq i},Y_\text{label})}{\displaystyle\sum_{Y'}P_{\theta}(X_{\leq i},Y')}
\end{aligned}
$$

Where:
- $X_{\leq i}$: Input sequence up to position $i$, representing the email content.
- $Y_\text{label}$: The label indicating whether the email is spam or not spam.
- $\theta$: Parameters of the pre-trained language model.
- $P_\theta(Y_\text{label}|X_{\leq i})$: Posterior probability of label $Y_\text{label}$ given input $X_{\leq i}$
- $P_\theta(X_{\leq i}|Y_\text{label})$: Likelihood of input $X_{\leq i}$ given label $Y_\text{label}$
- $P_\theta(X_{\leq i},Y_\text{label})$: Joint probability of input $X_{\leq i}$ and label $Y_\text{label}$
- $P_\theta(Y_\text{label})$: Prior probability of label $Y_\text{label}$

### Quick Start Checklist

1. Python Environment
   - [Install UV](https://docs.astral.sh/uv/getting-started/installation/#installing-uv).
   - Install Python dependencies with the following command from the project root:
    ```bash
    uv sync
    ```
2. Git clone autograder to this directory.
    ```bash
    git clone git@github.com:DSL-Lab/CPEN455-Project-2025W1-Autograder.git autograder
    ```
    If you are using windows laptop, you should first run `git clone git@github.com:DSL-Lab/CPEN455-Project-2025W1-Autograder.git` and then rename it to `autograder`.
    - Confirm the released datasets exist under `autograder/cpen455_released_datasets/`

3. Run the provided examples:
    ```
    # Run chatbot example
    uv run -m examples.chatbot_example 
    ```
    ```
    # Run bayes inverse zero shot example
    bash examples/bayes_inverse_zero_shot.sh
    ```
    ```
    # Run bayes inverse naive prompting example
    bash examples/bayes_inverse_naive_prompting.sh
    ```
    ```
    # Run bayes inverse full finetune example
    bash examples/bayes_inverse_full_finetune.sh
    ```

4. Validate outputs:
    - Generate probabilities on test set with the following command (writes CSVs into `bayes_inverse_probs/`):
    ```bash
    uv run -m examples.save_prob_example
    ```
    - Validate everything end-to-end by executing `bash autograder/auto_grader.sh`; this is the same entry point used during grading.

### Other Methods to Explore
If you find that simply utilizing the provided examples does not yield satisfactory results, any methods are welcome. Here are some suggestions if you need inspiration:

**Prefix-Tuning**: Instead of fine-tuning the entire model, you can explore prefix-tuning techniques where only a small set of parameters (prefix tokens) are trained while keeping the rest of the model fixed. This can be more efficient and may lead to better generalization.

**LoRA (Low-Rank Adaptation)**: LoRA introduces low-rank matrices to the model's weights, allowing for efficient fine-tuning with fewer parameters.

**Data Synthesis**: If you find the dataset too small, you can explore data synthesis techniques to generate additional training data via powerful LLMs. Ask them to generate synthetic spam and non-spam emails to augment your training set.

**Ensemble Methods**: Combine predictions from multiple models or different configurations of the same model to improve classification performance.

## ⭐️ Submission Instructions

### Preparing Your Submission
Submit your codebase and include a report.pdf file in the root directory of your project. Compress the whole project folder into a single zip file and submit it on Canvas.

File Tree For Important Files:
```
.
├── examples
│   ├── save_prob_example.py (The most important interface used for grading)
│   ├── {$your_training_code}.py (You should submit your training code if you trained a model)
│   └── ckpts
│       └── {$your_model}.ckpt(The model checkpoints you saved after training)
├── {$your_report}.pdf (Your project report in PDF format)
└── ... (All other necessary code files for your model's inference and training)
```

### Packaging Checklist
- Keep exactly one best checkpoint in `examples/ckpts/`; remove redundant checkpoints to reduce submission size.
- Verify `report.pdf` (NeurIPS style) lives in the repository root and references all experiments you ran.
- Exclude large caches when creating the archive (for example: `zip -r cpen455_project.zip . -x 'cache/*' 'wandb/*' '__pycache__/*'`).
- Run `bash autograder/auto_grader.sh` one last time before zipping so the graders see up-to-date outputs.

### ⭐️ Notes on Grading Interface

1. **`examples/save_prob_example.py` is the most important interface used for grading.**

    `examples/save_prob_example.py` is the most important interface used for grading. During grading, we will run `autograder/auto_grader.sh`, which is hardcoded to call `examples/save_prob_example.py` and then evaluate the results. 

    So please make sure you can run `autograder/auto_grader.sh` successfully. More details of autograder.sh can be found in the comments of `autograder/auto_grader.sh`.

    All of our grading related to accuracy will be based on the results from `autograder/auto_grader.sh`. The provided leaderboard in Kaggle is just for your reference and will not be used for grading.

2. **Only include one ckpt file if you trained a model.**

    Do not include all of your ckpt files in your submission. Only include the best one. If you include multiple ckpt files, ***we can't guarantee*** we will check all of them and ***pick the best one***.

3. Don't change any code in the `autograder` folder.

    The code in the `autograder` folder is used for grading, and we will use the git clone the original code from github to the `autograder` folder for grading. 

    If you change any code in the `autograder` folder, ***we can't guarantee*** autograder will work properly. And this may lead to a ZERO grade related to accuracy performance.


## ⭐️ Grading Breakdown

| Milestone | Deliverable / Evidence | How to Verify / Run | Weight |
|-----------|-----------------------|---------------------|--------|
| Chatbot Example | Run the chatbot with several prompts and summarize behaviour in the report | `uv run -m examples.chatbot_example` | 10% |
| Bayes Inverse Zero Shot | Produce zero-shot predictions and discuss accuracy | `bash examples/bayes_inverse_zero_shot.sh` | 10% |
| Bayes Inverse Naive Prompting | Compare prompted vs. zero-shot performance in the report | `bash examples/bayes_inverse_naive_prompting.sh` | 10% |
| Bayes Inverse Full Finetune | Launch the fine-tuning loop and report results | `bash examples/bayes_inverse_full_finetune.sh` | 10% |
| Accuracy ≥ 80% | Achieve at least 80% accuracy on the test set | submit to kaggle leaderboard | +5% |
| Accuracy ≥ 85% | Achieve at least 85% accuracy on the test set | submit to kaggle leaderboard | +5% |
| KV Cache Explanation | Explain decoder-only models, KV cache usage, drawbacks, and repo implementation | Report section referencing `model/` and `utils/` code | 20% |
| Leaderboard Competition | Make sure autograder works | All students' performance evaluated and ranked by TAs | Up to 30% |

### ⭐️ Basic Parts With Code Examples (40% of total grade)

#### Run the provided code examples and write the corresponding reports:
##### 1.`examples/chatbot_example.py`(10% of total grade)
+ 5% of total grade for running, as long as you try different prompts and show the outputs. 
+ 5% for report

##### 2.`examples/bayes_inverse_zero_shot.sh`(10% of total grade)
+ 5% of total grade for running, as long as you get the results.
+ 5% for report

##### 3.`examples/bayes_inverse_naive_prompting.sh`(10% of total grade)
+ 5% of total grade for running, as long as you get the results.
+ 5% for report

##### 4.`examples/bayes_inverse_full_finetune.sh`(10% of total grade)
+ 5% of total grade for running, as long as you can launch and train the model successfully.
+ 5% for report

### ⭐️ Advanced Parts (60% of total grade)

#### Achieve over 80% Accuracy (5% of total grade)
+ As long as you can achieve over 80% accuracy on the test set using any method, you will get full 5% of total grade for this part.

+ As long as you can achieve over 80% accuracy on the test set using any method, just ignore the first part, you will get full marks for this part. i.e., 40% of total grade:

    + [Chatbot Example ~ 10%](#1exampleschatbot_examplepy10-of-total-grade)
    + [Bayes Inverse Zero Shot Example ~ 10%](#2examplesbayes_inverse_zero_shotsh10-of-total-grade)
    + [Bayes Inverse Naive Prompting Example ~ 10%](#3examplesbayes_inverse_naive_promptingsh10-of-total-grade)
    + [Bayes Inverse Full Finetune Example ~ 10%](#4examplesbayes_inverse_full_finetunesh10-of-total-grade)

#### Achieve over 85% Accuracy (5% of total grade)
+ As long as you can achieve over 85% accuracy on the test set using any method, you will get 5% of total grade for this part.

#### Explain KV Cache Mechanism (20% of total grade)
+ 5% of total grade for explain what is decoder-only transformers.
+ 5% for explain the KV Cache mechanism in decoder-only transformers, why it is useful and how it works.
+ 5% for explain what is the drawback of using KV Cache.
+ 5% for explain how KV Cache is implemented in our provided codebase.

#### Leaderboard Competition (30% of total grade)
The remaining 30% of total grade will be based on the performance among all students in this class.

If you achieved N-th place, you will get $(1 - \frac{N}{\text{Number of students}}) \times 30$ % of total grade for this part.

## Final project report guidelines
Students are required to work on projects individually. All reports must be formatted according to the NeurIPS conference style and submitted in PDF format. The official template can be downloaded from [the NeurIPS style files page](https://neurips.cc/Conferences/2023/PaperInformation/StyleFiles). When combining the report with source code and additional results, the submission on Canvas portal should be in a zip file format.


### Report Length and Structure:

+ The report should not exceed 4 pages, excluding references or appendices.
+ A lengthy report does not equate to a good report. Don't concern yourself with the length; instead, focus on the coding and consider the report as a technical companion to your code and ideas. We do not award extra points for, nor penalize, the length of the report, whether it be long or short.
+ We recommend organizing the report in a three-section format: Model, Experiments, and Conclusion.

### Model Presentation Tips:

+ Include a figure (created by yourself!) illustrating the main computation graph of the model for better clarity and engagement.
+ Use equations rigorously and concisely to aid understanding.
+ An algorithm box is recommended for clarity if the method is complex and difficult to understand from text alone.
+ Provide a formal description of the models, loss functions etc.
+ Distinguish your model from others using comparative figures or tables if possible.
### Experiments Section:
Including at least one of the following is recommended:
+ Ablation study focusing on specific design choices.
+ Information on training methods, and any special techniques used.
+ Both quantitative and qualitative analysis of experimental results.
### Conclusion Section:
+ Summarize key findings and contributions of your project.
+ Discuss limitations and potential avenues for future improvements and research.

### Citation and Disclosure Requirements:
+ Cite every external dataset, paper, blog post, or code snippet that informs your work.
+ Document any AI-assisted coding (e.g., ChatGPT, Copilot) by including prompt-response summaries in an appendix and mentioning the assistance in the main text.
+ Credit collaborators for informal feedback and describe any reused internal code bases.

## Academic Integrity Guidelines for the Course Project

In developing your model, you are permitted to utilize any functions available in PyTorch and to consult external resources. However, it is imperative to properly acknowledge all sources and prior work utilized.

Violations of academic integrity will result in a grade of ZERO. These violations include:

1. Extensive reuse or adaptation of existing methods or code bases without proper citation in both your report and code.
2. Use of tools like ChatGPT or Copilot for code generation without proper acknowledgment, including details of prompt-response interactions.
3. Deliberate attempts to manipulate the testing dataset in order to extract ground-truth labels or employ discriminative classifiers.
4. Intentional submission of fabricated results to the competition leaderboard.
5. Any form of academic dishonesty, such as sharing code, model checkpoints, or inference results with other students.

Adhering to these guidelines is crucial for maintaining a fair and honest academic environment. Thank you for your cooperation.
