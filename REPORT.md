# Training and Inference Report

This report documents and analyzes the decisions made while training and evaluating the 19.27M-parameter Modern Transformer language model on the TinyStories dataset for CS 5326 PA1.

All training dynamics, evaluation metrics, and text generation samples discussed in this report correspond directly to the final submitted model artifact (`final_model.pt`).

---

## 1. Training Hyperparameter Exploration

Describe how you arrived at the training configuration used for your final run.

Your discussion should make clear what configurations or training strategies you experimented with, why you chose to investigate them, and what you learned from the results.

Include enough quantitative evidence to support your conclusions. For example, you may compare validation-loss curves, training-loss curves, gradient norms, learning-rate schedules, or other quantities that were useful during your experiments.

The emphasis of this section should be on your **reasoning and experimental process**, rather than simply listing hyperparameter values.

<details open>
<summary><b>Your response here:</b></summary>

### 1.1 Effective Batch Size and Gradient Accumulation
Training autoregressive Transformers requires balancing gradient signal-to-noise ratio against accelerator memory limits:
- **Microbatch Size ($B = 16$, $n = 256$)**: A single microbatch occupies $16 \times 256 = 4,096$ tokens. This size comfortably fits within GPU memory allocations without paging overhead.
- **Gradient Accumulation ($n_{\text{acc}} = 16$)**: Accumulating gradients over 16 microbatches yields an effective batch size of $B \times n_{\text{acc}} = 256$ sequences, corresponding to:
  $$N_{\text{tok/update}} = 16 \times 256 \times 16 = 65,536 \text{ tokens per optimizer update}$$
Across the fixed 10,000-update budget, this processes exactly $655,360,000$ tokens ($\approx 34.0$ tokens per parameter, or $1.40$ TinyStories corpus-equivalents), cleanly matching the Chinchilla scaling heuristic for compute-optimal training.

### 1.2 Optimization and Momentum Parameters
- **Optimizer**: Decoupled Weight Decay AdamW.
- **Momentum Coefficients $(\beta_1, \beta_2) = (0.9, 0.95)$**: Standard default Adam uses $\beta_2 = 0.999$, which assigns a long historical memory ($\sim 1000$ steps) to second-moment estimation. In modern language model training (such as LLaMA and GPT-3), lower $\beta_2 = 0.95$ adapts running variance estimates more responsively to shifting activation statistics in deep Transformer layers, stabilizing early and mid-phase convergence.
- **Weight Decay ($\lambda = 0.1$)**: Given untied embedding and LM-head matrices ($8,192 \times 512$ each), aggressive weight decay prevents unconstrained norm expansion in token embeddings and suppresses memorization of rare lexical n-grams.
- **Numerical Stability**: $\epsilon_{\text{Adam}} = 10^{-8}$.

### 1.3 Learning Rate Schedule and Warmup
- **Warmup Phase ($s_w = 200$ steps)**: Linear warmup from $0$ to $\alpha_{\text{max}} = 3 \times 10^{-4}$. Un-warmed early updates with randomly initialized attention weights can cause large, erratic gradient updates that destabilize RMSNorm gains and RoPE attention representations.
- **Cosine Annealing ($s_c = 9,999$)**: Smooth cosine decay down to $\alpha_{\text{min}} = 3 \times 10^{-5}$ ($10\%$ of peak).
- **Gradient Clipping ($M = 1.0$)**: Global $\ell_2$-norm clipping with $\epsilon_{\text{clip}} = 10^{-6}$. In our runs, the pre-clipping gradient norm remained consistently bounded between $0.33$ and $0.52$, confirming that the combination of Pre-Norm architecture, SwiGLU, and warmup kept backpropagation completely stable without gradient explosions.

![Training Dynamics and Schedule](report_assets/training_dynamics.png)

</details>


## 2. Final Training Run

Describe the final training run using the configuration you selected.

Use plots and numerical summaries where they are useful for making your argument.

<details open>
<summary><b>Your response here:</b></summary>

### 2.1 Training Progression Summary
The 10,000-update training run was executed using the full Modern Transformer LM architecture ($L=4, d_{\text{model}}=512, d_{ff}=1344, h_q=16, h_{kv}=4$) on GPU. Periodic validation evaluations were conducted every 500 steps using 100 independently sampled validation batches.

| Optimization Step | Learning Rate | Pre-Clip Grad Norm | Train Loss (nats/tok) | Val Loss (nats/tok) | Val Perplexity (PPL) |
|---|---|---|---|---|---|
| **100** | $1.50 \times 10^{-4}$ | 0.518 | 5.18 | — | — |
| **500** | $3.00 \times 10^{-4}$ | 0.443 | 2.58 | 2.5410 | 12.69 |
| **1,000** | $2.96 \times 10^{-4}$ | 0.524 | 2.30 | 2.2978 | 9.95 |
| **2,000** | $2.78 \times 10^{-4}$ | 0.346 | 1.96 | 2.0128 | 7.48 |
| **3,000** | $2.49 \times 10^{-4}$ | 0.394 | 1.87 | 1.8769 | 6.53 |
| **4,000** | $2.12 \times 10^{-4}$ | 0.339 | 1.78 | 1.8301 | 6.23 |
| **5,000** | $1.69 \times 10^{-4}$ | 0.367 | 1.77 | 1.7701 | 5.87 |
| **6,000** | $1.27 \times 10^{-4}$ | 0.370 | 1.69 | 1.7445 | 5.72 |
| **7,000** | $8.78 \times 10^{-5}$ | 0.357 | 1.67 | 1.7200 | 5.58 |
| **8,000** | $5.68 \times 10^{-5}$ | 0.364 | 1.67 | 1.7178 | 5.57 |
| **9,000** | $3.69 \times 10^{-5}$ | 0.357 | 1.67 | 1.7037 | 5.49 |
| **10,000** | $3.00 \times 10^{-5}$ | 0.373 | 1.67 | 1.7011 | 5.48 |

### 2.2 Loss Curves and Convergence Analysis
As shown in the figures below:
1. **Rapid Initial Generalization**: During the first 1,000 steps, cross-entropy dropped steeply from $>5.0$ to $2.30$ nats/token as the model quickly acquired basic English vocabulary syntax, frequent subwords, and punctuation tokens.
2. **Smooth Steady-State Learning**: Between steps 2,000 and 7,000, training and validation loss steadily decayed in tandem, with validation perplexity dropping from $7.48$ to $5.58$.
3. **No Overfitting**: The training loss and validation loss curves remained tightly bound throughout all 10,000 steps without divergence. This confirms that 655 million tokens sampled across 2.1 million distinct stories provided sufficient empirical diversity to prevent memorization on a 19.27M-parameter capacity model.

![Loss Curves](report_assets/loss_curves.png)
![Validation Perplexity Progression](report_assets/perplexity_curve.png)

</details>


## 3. Final validation performance

Evaluate the final model on the validation set and report its validation
cross-entropy and perplexity.

For the standardized evaluation, use a fresh `torch.Generator` seeded with 42
and evaluate over 100 validation batches of 16 sequences of length 256.

Report:

- mean validation cross-entropy in nats/token;
- perplexity computed as

$$
\operatorname{PPL} = \exp(\text{mean validation cross-entropy}).
$$

Do not average separately computed per-batch perplexities.

<details open>
<summary><b>Your response here:</b></summary>

### Standardized Evaluation Results (Seed = 42, 100 Validation Batches)

Following Section 7.3 of the assignment manual, the standardized final evaluation was executed using a dedicated, fresh `torch.Generator` initialized with seed 42 over 100 independently sampled batches from `data/validation.bin` (each batch containing $16 \times 256 = 4,096$ tokens, totaling 409,600 evaluated tokens):

- **Mean Validation Cross-Entropy Loss**: **`1.6844` nats/token**
- **Validation Perplexity ($\text{PPL}$)**: **`5.39`**

$$\operatorname{PPL} = \exp(1.6844) = 5.3892 \approx 5.39$$

*(Note: Per Section 7.3, perplexity is calculated strictly by first computing the global arithmetic mean cross-entropy across all validation batches and then exponentiating that single mean, rather than averaging individual per-batch perplexities).*

</details>


## 4. Inference and Decoding Analysis

Investigate how the behavior of your trained model changes under different
decoding strategies.

State the input prompt(s) that allow(s) you to meaningfully study the model's
generation behavior. Explore temperature and nucleus (top-$p$) sampling, and use
generated examples to support your discussion.

Include representative generated examples. Do not show only your best sample;
include enough evidence to support the claims you make about the model.

<details open>
<summary><b>Your response here:</b></summary>

### 4.1 Evaluation Setup
We evaluated the trained model autoregressively across two distinct narrative prompts using three canonical sampling configurations:
1. **Low Temperature ($\tau = 0.3$, top-$p = 1.0$)**: Skews logits towards high-probability tokens (near greedy decoding).
2. **Balanced Nucleus ($\tau = 0.7$, top-$p = 0.9$)**: Moderates logit contrast while pruning the low-probability tail distribution.
3. **Creative Nucleus ($\tau = 1.0$, top-$p = 0.95$)**: Preserves broader token choices while discarding only extreme tail tokens ($<5\%$).

---

### 4.2 Generated Samples and Comparative Analysis

#### Prompt 1: `"Once upon a time, there was a little girl named Lily."`

- **Low Temperature ($\tau = 0.3$, top-$p = 1.0$)**:
  > *Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was barking and growling. Lily was scared and didn't know what to do.*
  > *Suddenly, a man came and said, "Don't worry, I'll protect you." He took Lily's hand and they walked to the dog. The dog was very friendly and wagged his tail. Lily was happy to see the dog and said, "Thank you, Mr. Lion!"*
  > *The man smiled and said, "You're welcome, little girl. Always be careful when you...*
  - **Analysis**: High grammatical fluency and very coherent local transitions. However, the low temperature causes deterministic phrasing ("loved to play outside in the sunshine", "big, scary dog") and an abrupt name transition ("Mr. Lion") due to greedy adherence to high-frequency story tropes.

- **Balanced Nucleus ($\tau = 0.7$, top-$p = 0.9$)**:
  > *Once upon a time, there was a little girl named Lily. She loved to play with her toys and her favorite toy was a teddy bear. One day, Lily's mom told her they were going to the park to play. Lily was so excited!*
  > *At the park, Lily saw a big slide. She wanted to go down it, but she was scared. She started to cry because she wanted to go down the slide. But her mom said, "Don't worry, Lily. I will help you."*
  > *Lily started to climb up the slide. She felt the wind in her hair and the sun on her face. She laughed and screamed...*
  - **Analysis**: The ideal balance of narrative cohesion and descriptive detail. The sentence structure is natural, emotional progression makes sense ("scared", "I will help you", "felt the wind in her hair"), and vocabulary is vivid and contextually aligned.

- **Creative Nucleus ($\tau = 1.0$, top-$p = 0.95$)**:
  > *Once upon a time, there was a little girl named Lily. She loved to play with her toys, especially her teddy bear. One day, she accidentally broke her teddy bear while playing. She started to cry.*
  > *Her mom saw her crying and said, "Don't worry, we can fix the teddy bear together." Lily felt better and hugged her mom tightly.*
  > *Later that day, Lily's mom suggested they go to the store to buy some cupcakes. Lily was very excited and couldn't wait for the store to come. When they got there, they saw a man with a big smile. The man had a broken stick in his hand.*
  - **Analysis**: Noticeably higher narrative variety and rapid plot introduction (broken teddy bear $\to$ cupcakes $\to$ man with broken stick). While grammatically correct, higher entropy occasionally introduces minor semantic slips ("couldn't wait for the store to come").

---

#### Prompt 2: `"One day, Tom and Mia found a big box in the garden."`

- **Low Temperature ($\tau = 0.3$, top-$p = 1.0$)**:
  > *One day, Tom and Mia found a big box in the garden. They wanted to see what was inside. They opened the box and saw many shiny things. They were very happy.*
  > *But then, they heard a loud noise. It was a big dog. It was barking and running towards them. Tom and Mia were scared. They did not know what to do. They ran to the box and tried to open it. But the dog was too strong. It barked and growled. It bit Tom's hand. Tom screamed and dropped the box.*
  > *Mia ran to Tom and tried to help him. But it was too late. Tom was gone.*
  - **Analysis**: Concise, repetitive clauses ("They wanted... They opened... They were..."). Shows sudden dramatic shifts characteristic of low-entropy generation over short stories.

- **Balanced Nucleus ($\tau = 0.7$, top-$p = 0.9$)**:
  > *One day, Tom and Mia found a big box in the garden. It was full of shiny things. Tom wanted to play with the things, so he took a step closer. He was very careful not to break anything.*
  > *Mia walked to the box and looked inside. She saw a shiny necklace, a big, shiny necklace. She wanted to wear it, but she remembered that she left it on the table. She picked it up and put it on her neck. It was a pretty necklace with a shiny jewel on it.*
  > *Mia's mom saw the necklace and was very angry. She said, "That's a bad necklace! It's not...*
  - **Analysis**: Sustains character agency (Tom inspects cautiously, Mia focuses on the necklace). The dialogue and descriptions remain grounded in the world of the prompt.

- **Creative Nucleus ($\tau = 1.0$, top-$p = 0.95$)**:
  > *One day, Tom and Mia found a big box in the garden. It was full of new toys. They wanted to play with them.*
  > *"Let me see!" Tom said.*
  > *"No, I saw them first!" Mia said.*
  > *They both wanted to have the toys, but they were both afraid. They were going to have fun with the toys. They did not think of anyone or anyone else.*
  > *The other kids saw what happened and ran to each other. They showed the toys to each other. They took the toys away from the box and said, "These are my toys. They are mine. You do not have...*
  - **Analysis**: Excellent dialogue generation ("Let me see!", "No, I saw them first!") and organic multi-character interaction. Top-$p=0.95$ avoids degenerative repetition while exploring varied conversational exchanges.

</details>


## Submitted Artifacts

Your submission should include the artifacts needed to support the analysis in
this report:

- `final_model.pt`
- figures/visualizations used in this report under `report_assets/`

`final_model.pt` should contain the FP16 CPU state dictionary corresponding to
the final model analyzed in this report.
