# Comparing Transformer Architectures and their Loss Functions using Music Data

**Authors:** Sahar Fatemi, Daniel Armani

---

## Abstract

We analyze and compare different transformer architectures especially regarding their loss function for their training efficiency and out of sample performance. We use music data as context to train and test the models experimentally.

---

## 1. Introduction

The Transformer architecture (Vaswani et al., 2017) has become the dominant backbone for sequence modeling across language, vision, and, increasingly, symbolic music generation. At the core of the Transformer is self-attention, which computes a weighted combination of value vectors using similarity scores between queries and keys, conventionally normalized with the softmax function. While softmax attention is expressive and well understood, it comes with a fundamental computational cost: computing and storing the full N×N attention matrix scales quadratically in both time and memory with sequence length N (Katharopoulos et al., 2020; Choromanski et al., 2021; Qin et al., 2022). This quadratic cost limits the practical sequence lengths that can be modeled without specialized engineering, and has motivated a substantial body of work on more efficient attention mechanisms.

One line of this work asks a more fundamental question: is softmax itself necessary? A growing family of methods replaces the softmax normalization with alternative similarity functions that admit linear-time and linear-memory formulations. Katharopoulos et al. (2020) show that self-attention can be expressed as a linear dot-product between kernel feature maps, revealing a formal equivalence between Transformers and recurrent networks and enabling constant-memory autoregressive inference. Choromanski et al. (2021) approximate softmax attention itself via random feature maps (FAVOR+), while other work departs from softmax entirely: Lu et al. (2024) replace the exponential similarity with a Gaussian kernel over query-key distances (SOFT), Mongaras et al. (2024) replace it with L2-normalized cosine similarity (Cottention), and Qin et al. (2022) combine a non-negative ReLU transform with a cosine-based positional reweighting scheme (CosFormer) that achieves state-of-the-art results on the Long-Range Arena benchmark without ever computing a softmax. Underlying much of this work is a shared theoretical picture: attention can be understood generally as kernel smoothing over a sequence (Tsai et al., 2019), of which softmax attention is only one particular instance. Recent theoretical work has further questioned whether the specific properties softmax enforces (non-negativity, row-normalization into a probability distribution, and sparsity) are in fact responsible for its empirical success, or whether its benefit instead comes from an implicit regularization effect on the norm of the attention matrix that other, non-probabilistic activations can replicate (Saratchandran et al., 2024). Together, these results suggest that removing softmax from attention is not merely an efficiency compromise, but a design space in its own right that remains incompletely characterized empirically.

Symbolic music offers a well-suited and comparatively underused domain for studying this design space. Following work that represents music as a sequence of discrete tokens amenable to language-model-style training (Huang et al., 2019; Huang and Yang, 2020; Hsiao et al., 2021), the effective vocabulary size of symbolic music representations is typically one to two orders of magnitude smaller than natural-language vocabularies; REMI-style encodings (Huang and Yang, 2020), for instance, represent pitch, duration, velocity, and metrical position with a few hundred discrete tokens in total. This makes it feasible to train Transformers with these tokenizations from scratch under tightly controlled conditions, without the confound of large-scale pretraining that dominates most existing evaluations of softmax-free attention (Choromanski et al., 2021; Mongaras et al., 2024; Qin et al., 2022). Symbolic music also has structural properties that make it a distinctive testbed for the loss-function question we study: long-range repetition that motivates relative rather than purely absolute positional representations (Huang et al., 2019), and, unlike most natural-language tokens, a substantial fraction of its vocabulary (pitch, duration, and velocity tokens) that is explicitly ordinal rather than categorical.

This ordinal structure is directly relevant to how these models are trained. The standard training objective for autoregressive sequence models, cross-entropy loss, treats all incorrect predictions as equally wrong regardless of their numerical proximity to the correct answer, an assumption well suited to nominal categories but not to quantities with a natural ordering (Terven et al., 2025); predicting a pitch one semitone away from the target incurs the same loss as predicting a pitch at the opposite end of the keyboard. Nath et al. (2024) show theoretically and experimentally that Transformer encoders are poor approximators of smooth functions, requiring an exponential number of layers to achieve fine-grained resolution, a limitation directly relevant to a loss that must implicitly learn numerical proximity from data alone. Zausinger et al. (2025) propose Number Token Loss (NTL), an auxiliary regression-style term added to cross-entropy that explicitly rewards numerical closeness on tokens with numerical meaning, reporting consistent improvements on mathematical reasoning tasks and performance competitive with a dedicated regression head, with the method scaling to models of up to 3B parameters. Symbolic music's ordinal pitch, duration, and velocity tokens are a natural test of whether this benefit generalizes beyond arithmetic tasks to a domain where numerical proximity carries distinct perceptual and musical meaning.

Despite this body of work, three gaps remain. First, existing softmax-free attention mechanisms have largely been evaluated in isolation, each validated on a different task or benchmark: vision classification for SOFT (Lu et al., 2024), masked and causal language modeling for Cottention (Mongaras et al., 2024), and the Long-Range Arena and GLUE for CosFormer (Qin et al., 2022). This leaves it unclear how they compare to one another under identical training conditions. Second, Timkey and van Schijndel (2021) show that cosine similarity and Euclidean distance in large pretrained language models (BERT, RoBERTa, GPT-2, XLNet) are dominated by a small number of "rogue dimensions," a finding with direct relevance to distance- and cosine-based attention mechanisms that compute exactly these quantities between queries and keys; yet, to our knowledge, this analysis has not been extended to smaller, from-scratch-trained models outside the text domain, where softmax-free attention is a direct architectural choice rather than a post-hoc approximation of a pretrained softmax model. Third, while Number Token Loss has been evaluated on general-purpose language and mathematical reasoning tasks, it has not, to our knowledge, been tested in combination with softmax-free attention mechanisms, nor on a domain where ordinal token structure is intrinsic to the data rather than incidental.

In this work, we address these gaps using symbolic piano music from the MAESTRO dataset as a controlled experimental context. We train decoder-only Transformers with REMI-style tokenization (Huang and Yang, 2020) from scratch, holding model size, data, and training procedure fixed, and vary two factors: the attention mechanism (Gaussian/SOFT-style, cosine/Cottention-style, and CosFormer-style, none of which use softmax) and the training loss (cross-entropy alone versus cross-entropy augmented with Number Token Loss). We ask three research questions: (RQ1) how do these three softmax-free attention mechanisms compare in training efficiency and out-of-sample performance when trained under identical conditions; (RQ2) does treating pitch, duration, and velocity as ordinal quantities via an auxiliary regression loss improve performance over cross-entropy alone; and (RQ3) do the rogue dimensions documented in large pretrained text models appear in these smaller, from-scratch-trained music Transformers, and if so, does correcting for them improve performance.

Our contributions are threefold. First, we present a controlled, repeated-seed comparison of three softmax-free attention mechanisms on a shared symbolic music task, evaluating both training efficiency and out-of-sample generalization to a held-out test set; we find that Gaussian and CosFormer attention perform statistically equivalently across training lengths and random seeds, while cosine attention consistently underperforms both. Second, we provide, to our knowledge, the first evaluation of rogue-dimensionality in Transformers trained from scratch on a non-text domain, together with a direct test of the standardization-based correction it motivates; we find token embeddings in these models to be close to isotropic, more closely resembling static word embeddings than the large pretrained contextual models in which rogue dimensions were originally documented, with only mild concentration in query and key representations and no measurable downstream benefit from correcting it. Third, we present a rigorous, seed-controlled evaluation of Number Token Loss combined with softmax-free attention, including a decomposition analysis that isolates the auxiliary loss term's contribution to the training objective from its effect on the model itself; we find that Number Token Loss provides no measurable benefit over cross-entropy alone, on either next-token prediction accuracy or the numerical-closeness objective it is designed to improve, at the model and data scale we study.

---

## 2. Related Work

### 2.1 Efficient and Softmax-Free Attention

The quadratic cost of softmax attention (Vaswani et al., 2017) has motivated several distinct strategies for reducing it. One strategy approximates softmax attention itself while preserving its output distribution. Choromanski et al. (2021) introduce FAVOR+, a positive orthogonal random feature scheme that estimates the softmax kernel with linear space and time complexity and provable accuracy guarantees, and show it can also approximate other attention kernels beyond softmax. Tsai et al. (2019) provide a complementary theoretical contribution, reformulating attention generally as a kernel smoother over the input sequence; under this view, softmax attention, and the approximations built on top of it, are specific instances of a broader family of kernel-based similarity functions.

A second strategy replaces softmax entirely rather than approximating it. Katharopoulos et al. (2020) express self-attention as a linear dot-product between kernel feature maps, which removes the need to materialize the full attention matrix and reveals a formal equivalence between autoregressive Transformers and recurrent networks with a fixed-size hidden state. This linear-attention formulation underlies several of the softmax-free mechanisms evaluated in this work. Lu et al. (2024) build on it to propose SOFT, which replaces the exponential similarity with a Gaussian kernel over query-key distances and uses a Nystrom approximation with an iterative Newton-Raphson pseudo-inverse to keep the resulting attention linear in sequence length, originally developed for vision transformers. Mongaras et al. (2024) instead use L2-normalized cosine similarity (Cottention), which permits negative attention weights unlike softmax, and show the resulting attention can be reformulated as a recurrent network with constant memory at inference time. Qin et al. (2022) argue that only two properties of softmax, non-negativity and a re-weighting scheme that concentrates attention locally, are needed for strong performance, and propose CosFormer, which enforces non-negativity with a ReLU transform and re-weights attention scores using a cosine function of relative position, achieving results competitive with or better than softmax attention on the Long-Range Arena benchmark. Saratchandran et al. (2024) offer a theoretical account of why such departures from softmax's classic properties (non-negativity, row-normalization, and sparsity) can still succeed, arguing that softmax's main empirical benefit is an implicit regularization of the attention matrix's Frobenius norm, an effect that polynomial and other non-probabilistic activations can reproduce directly.

A third line of work addresses a limitation shared by most linear-attention formulations: because they avoid computing the explicit N by N attention matrix, they are not directly compatible with relative positional encoding schemes that require it. Luo et al. (2021) show that kernelized attention with relative positional encoding can be computed in O(n log n) time using the fast Fourier transform, since the relative position correlation matrix is Toeplitz, and additionally find that incorporating relative position information stabilizes training of kernelized attention against the large query and key norms that otherwise cause optimization instability. Liutkus et al. (2021) propose Stochastic Positional Encoding, the first relative positional encoding scheme compatible with O(N) linear-complexity Transformers, and evaluate it on both the Long-Range Arena benchmark and symbolic music generation. Neither mechanism is used in the present work, but both are directly relevant to the attention mechanisms we study, since Cottention and CosFormer are linear-complexity and currently rely on absolute positional encoding alone.

### 2.2 Symbolic Music Generation with Transformers

Applying Transformers to symbolic music generation requires converting a musical score into a token sequence, and several representations have been proposed for this purpose. Huang and Yang (2020) introduce REMI, a beat-based representation that explicitly encodes bar and metrical position events alongside note pitch, duration, and velocity, and show it produces music with clearer rhythmic regularity than earlier MIDI-like representations that leave metrical structure implicit. Hsiao et al. (2021) instead group related tokens describing a single musical event into a compound word, reducing sequence length and allowing a linear Transformer backbone (Katharopoulos et al., 2020) to generate full-length pop songs with type-specific prediction heads for each token category. Both representations address the same underlying challenge: symbolic music sequences are long relative to natural language, and Transformer self-attention scales poorly with sequence length.

A second challenge specific to music generation is capturing repetition and structure across long time spans. Huang et al. (2019) show that self-attention with a relative positional encoding, in which attention is modulated by the distance between positions rather than their absolute values, is essential to generating music with coherent long-term structure, and introduce an algorithmic improvement that reduces the memory cost of relative attention from quadratic to linear in sequence length. Yu et al. (2022) address long-range structure differently, combining fine-grained attention to a small set of structurally similar bars, selected by similarity statistics, with coarse-grained attention to a summary of all other bars, allowing Museformer to model sequences several times longer than full attention permits while producing more structured output.

Beyond representation and attention design, other work has explored alternative training objectives and conditioning strategies for music Transformers. Zhang (2023) combines adversarial training with a self-attention generator, using a reward signal computed directly from the discriminator rather than costly Monte Carlo rollouts, to encourage more musical long-sequence generation than teacher forcing alone produces. Shih et al. (2023) propose theme-conditioned generation, in which a Transformer encoder represents a musical theme retrieved via contrastive learning and clustering, and a decoder cross-attends to this representation so that the generated material develops the given theme rather than merely continuing it as a prompt. Both approaches use standard softmax attention and modify the training objective or conditioning mechanism rather than the attention computation itself, and are therefore orthogonal to the comparison in this work.

### 2.3 Loss Functions for Ordinal and Numerical Tokens

Terven et al. (2025) survey loss functions and metrics across deep learning tasks and note that cross-entropy, the standard objective for classification and next-token prediction, assumes a nominal scale over its output classes, treating every incorrect prediction as equally wrong regardless of how close it is to the correct answer, in contrast to regression losses that are designed to respect this kind of ordering. Nath et al. (2024) provide a theoretical grounding for why this matters for Transformers specifically, proving that Transformer encoders approximate smooth functions only as piecewise-constant functions whose resolution improves at best linearly with an exponential increase in the number of layers, and confirming experimentally that the full Transformer architecture shares this limitation.

Zausinger et al. (2025) propose Number Token Loss (NTL) as a direct response to this mismatch, adding a regression-style term to cross-entropy that is computed purely from token logits and requires only a mapping from number tokens to their numerical values. Two variants are proposed: NTL-MSE, which takes the mean squared error between the expected numerical value under the predicted token distribution and the true value, and NTL-Wasserstein, which instead measures the Wasserstein-1 distance between the predicted and true distributions over numerical values. The authors report that NTL consistently improves performance on mathematical reasoning tasks across model architectures, matches the performance of a dedicated regression head on a direct regression task despite operating purely at the token level, does not measurably hurt performance on text-only tasks, and scales successfully to models with up to 3 billion parameters. This work adopts the NTL-MSE variant, applied to the ordinal pitch, duration, and velocity tokens in our symbolic music vocabulary.

### 2.4 Representational Geometry in Transformers

A separate line of work examines the internal geometry of trained Transformer representations rather than their attention mechanism or training objective. Timkey and van Schijndel (2021) analyze cosine similarity and Euclidean distance between token representations in BERT, RoBERTa, GPT-2, and XLNet, and find that both measures are dominated by a small number of rogue dimensions, sometimes as few as one, regardless of pretraining objective; in the most extreme case they report, a single dimension in the final layers of XLNet accounts for more than 99 percent of the expected cosine similarity between randomly sampled token pairs. They further show that static, non-contextual embeddings such as Word2Vec and GloVe do not exhibit this pattern, and that standardizing the embedding space, subtracting the mean and dividing by the standard deviation per dimension, restores closer agreement between cosine similarity and human similarity judgments. This finding is directly relevant to any attention mechanism that computes cosine similarity or Euclidean distance between queries and keys, since rogue dimensions would cause such a mechanism to attend based on a small, arbitrary subset of the representation rather than its full content.

Razzhigaev et al. (2024) study a related but distinct property, the linearity of the transformation between consecutive layers in Transformer decoders, and find it to be near-perfect (a Procrustes similarity score of 0.99) across a range of open-source models, an effect they trace to the low norm each layer's contribution makes to the residual stream. They show that pruning or linearly approximating the most linear layers has little effect on model performance, and, more surprisingly, that adding a cosine-similarity-based regularization term between consecutive layer embeddings during pretraining reduces this linearity while improving downstream performance on SuperGLUE and TinyStories. Taken together with the rogue-dimensions finding, this work suggests that the internal representations of Transformer decoders are, by default, both less isotropic and less non-linear across depth than their large parameter counts might suggest, a consideration relevant to any architecture, including the shallow models used in this work, in which representational capacity cannot be taken for granted.

---

## 3. Method

### 3.1 Task and Data Representation

We train autoregressive, decoder-only Transformers for next-token prediction on symbolic piano music from the MAESTRO v3.0.0 dataset, which contains aligned MIDI transcriptions of solo piano performances with expressive timing and dynamics. Each MIDI file is converted into a token sequence following the REMI representation of Huang and Yang (2020), in which a new BAR token marks the start of each measure, a POSITION token places each note on a sixteenth-note grid within the bar, and each note is encoded as a VELOCITY, NOTE_ON (pitch), and DURATION token in sequence. Tempo is estimated from the first tempo change in each file, defaulting to 120 beats per minute where unavailable, and duration is quantized to sixteenth-note multiples. Our vocabulary omits the tempo and chord events used in the original REMI representation for simplicity.

The resulting vocabulary contains 196 tokens, summarized in Table 1. Three token types (NOTE_ON, DURATION, and VELOCITY) are ordinal: their token identity corresponds directly to a numerical quantity, and adjacent token identities are musically similar. The remaining token types (PAD, BOS, EOS, BAR, and POSITION) are structural and carry no numerical ordering.

Table 1: REMI-style vocabulary used in this work.

| Token type | Count | Token IDs | Meaning |
|---|---|---|---|
| PAD | 1 | 0 | Padding for batching |
| BOS | 1 | 1 | Beginning of sequence |
| EOS | 1 | 2 | End of sequence |
| BAR | 1 | 3 | Start of a new bar |
| POSITION | 16 | 4-19 | Position on a sixteenth-note grid within a bar |
| NOTE_ON | 128 | 20-147 | MIDI pitch (0-127) |
| DURATION | 16 | 148-163 | Note duration in sixteenth-note multiples (1-16) |
| VELOCITY | 32 | 164-195 | Quantized MIDI velocity bin |
| **Total** | **196** | | |

Each full-length token sequence is split into overlapping fixed-length windows of 512 tokens with 50 percent stride, forming (input, target) pairs for next-token prediction.

### 3.2 Model Architecture

Decoder-only Transformer (Vaswani et al., 2017), pre-norm formulation: x + Attention(LayerNorm(x)), then x + FeedForward(LayerNorm(x)), GELU feedforward, fixed sinusoidal positional encoding added once at the input (no relative positional encoding, unlike Huang et al. 2019, a choice revisited in the Discussion), weight-tied output projection, Xavier uniform init.

Table 2: Model hyperparameters, held fixed across all experiments.

| Hyperparameter | Value |
|---|---|
| Embedding dimension (d_model) | 64 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Feedforward dimension (d_ff) | 256 |
| Dropout | 0.1 |
| Sequence length | 512 |
| Vocabulary size | 196 |
| Trainable parameters | 112,128 to 112,136 (mechanism-dependent) |

### 3.3 Attention Mechanisms

We compare three attention mechanisms, each substituted for the standard scaled dot-product softmax attention while keeping the surrounding architecture described in Section 3.2 fixed. None of the three applies a softmax normalization at any point; each instead replaces it with an alternative similarity function and normalization scheme, following the softmax-free attention literature reviewed in Section 2.1.

Gaussian attention follows the SOFT formulation of Lu et al. (2024), in which the similarity between query i and key j is given by a Gaussian kernel over their distance, exp(-||Q_i - K_j||^2 / 2*sqrt(d_head)), rather than the exponentiated dot product used by softmax attention. Under this formulation, queries and keys that are close in embedding space receive an attention weight near one, while distant pairs receive a weight near zero, and the resulting scores are normalized by their row sum rather than by a softmax. Lu et al. (2024) introduce this Gaussian kernel primarily to enable a Nystrom-based low-rank approximation that keeps attention linear in sequence length, which is the central computational contribution of the SOFT paper and is intended for the long token sequences common in vision transformers. In our setting, sequence length is fixed at 512 tokens throughout, and we therefore always compute the exact formulation of the kernel over the full sequence rather than the Nystrom approximation. We state this explicitly because it means that every Gaussian attention result reported in this paper reflects the exact, quadratic-complexity kernel, not the linear-complexity approximation that SOFT proposes; our results should accordingly be read as an evaluation of the Gaussian similarity function itself, not of the efficiency mechanism built on top of it.

Cosine attention follows the Cottention formulation of Mongaras et al. (2024). Queries and keys are first L2-normalized, so that their dot product becomes a cosine similarity bounded in the range negative one to one. This is a substantive departure from softmax attention, which produces only non-negative weights: cosine attention can assign a negative weight to a key, effectively subtracting its value vector from the output rather than merely down-weighting it. Because cosine similarity does not sum to one across keys the way softmax probabilities do, a learned per-head scalar is used to rescale the value vectors and keep the resulting attention magnitudes stable as sequence length grows. For causal, autoregressive training, we use the cumulative-sum reformulation described by Mongaras et al. (2024), which expresses the causal attention output as a running sum of outer products between keys and values. This avoids materializing an explicit attention matrix and keeps the computation linear in sequence length.

CosFormer attention follows Qin et al. (2022), who argue that only two properties of softmax attention are needed to match its performance. The first, non-negativity, is enforced here by passing queries and keys through a ReLU rather than by an exponential and normalization step. The second, a re-weighting scheme that biases attention toward nearby positions, is implemented as a cosine function of relative position. Because this cosine re-weighting decomposes into a sum of two terms through the Ptolemy identity, it can be applied without computing an explicit attention matrix, in the same manner as cosine attention above, and we use the corresponding causal, cumulative-sum formulation for autoregressive training.

### 3.4 Q/K Standardization Ablation

In addition to the three attention mechanisms described above, we implement one further modification as a separate, optional ablation, and not as part of the main model definition used elsewhere in this paper. Timkey and van Schijndel (2021) show that cosine similarity and Euclidean distance in large pretrained language models can be dominated by a small number of rogue dimensions, and that standardizing the representation before computing either quantity, subtracting its mean and dividing by its standard deviation per token, corrects for this. Since Gaussian attention computes a Euclidean distance and cosine attention computes a cosine similarity directly between queries and keys, this correction is directly applicable to both mechanisms. We therefore add an optional standardization step, applied to the query and key projections immediately after they are computed and before the attention mechanism itself, for Gaussian and cosine attention only. CosFormer is excluded from this ablation, since its ReLU and cosine re-weighting operations do not compute a raw distance or cosine similarity between queries and keys in the same sense as the other two mechanisms. This standardization step is disabled by default, and every result reported in this paper other than the standardization ablation itself in Section 5 uses the unmodified query and key projections described in Section 3.3.

### 3.5 Loss Functions

All models are trained with next-token prediction, and we compare two training objectives applied to this same underlying task.

The first, cross-entropy (CE), is the standard objective for autoregressive sequence models: the cross-entropy between the predicted distribution over the vocabulary and the true next token, computed over all non-padding positions. As discussed in Section 2.3, cross-entropy treats every incorrect prediction as equally wrong regardless of how numerically close it is to the correct token, an assumption that does not reflect the ordinal structure present in our NOTE_ON, DURATION, and VELOCITY tokens.

The second objective augments cross-entropy with Number Token Loss (NTL), following the NTL-MSE formulation of Zausinger et al. (2025). For positions where the target token is ordinal, we compute the expected numerical value under the model's predicted token distribution, sum_i p_i * value_i, where value_i is the normalized numerical value associated with token i, taken to be zero for structural tokens, and take the mean squared error between this expected value and the true value. Pitch, duration, and velocity values are each linearly normalized to the range zero to one before this computation, so that all three ordinal token types contribute to the loss on a comparable scale. The resulting auxiliary term, L_NTL, is added to the standard cross-entropy loss with a fixed weight, L = L_CE + lambda * L_NTL, with lambda = 0.5 in every experiment reported in this paper.

### 3.6 Rogue-Dimension Diagnostic

To test whether the rogue dimensions documented by Timkey and van Schijndel (2021) in large pretrained text models appear in our smaller, from-scratch-trained music Transformers, we implement a read-only diagnostic that closely follows their methodology. For a trained model, we extract five representation types: the token embedding matrix, the hidden state after each Transformer layer, and the query and key projections at each layer, the latter at two granularities, both before the multi-head split (the full d_model-dimensional representation) and after it (the d_head-dimensional per-head representation, which is the space in which Gaussian and cosine attention actually compute their similarity scores).

For each representation matrix X, containing N token vectors of dimension d, we compute three per-dimension quantities. The first is simply the variance of dimension i across all N tokens, Var_i(X). The second and third quantities follow Timkey and van Schijndel's Equations 3 and 4, which define the contribution of dimension i to the cosine similarity of a pair of representations (u, v) as CC_i(u,v) = u_i * v_i / (||u|| * ||v||). We sample 3,000 token pairs (u, v) from X uniformly at random and estimate each dimension's typical contribution to cosine similarity as the mean of its absolute per-pair contribution, mean(|CC_i(u,v)|), rather than the mean of its signed contribution as in the original formulation. We take the absolute value before averaging because a dimension can contribute large positive and negative amounts across different pairs that would otherwise cancel in a signed average and mask real influence on the similarity computation; this is a deliberate, conservative adaptation of Timkey and van Schijndel's diagnostic for our purposes, not a direct reproduction of it. Analogously, for squared Euclidean distance, we compute each dimension's average squared difference across the same sampled pairs, mean((u_i - v_i)^2).

For all three quantities, variance, mean absolute cosine contribution, and mean squared distance contribution, we rank dimensions in descending order and report the proportion of the total attributable to the top one, three, and five dimensions. We compare each of these top-k proportions to the uniform baseline that would hold if no dimension were disproportionately influential, k / d for a d-dimensional representation. A top-k share substantially above this baseline indicates that those k dimensions are disproportionately responsible for the corresponding quantity, the signature of a rogue dimension in Timkey and van Schijndel's sense.

---

## 4. Experimental Setup

### 4.1 Training Procedure

All models are trained with the AdamW optimizer, with betas 0.9 and 0.98 and weight decay 0.01, and the learning rate schedule of Vaswani et al. (2017), which increases linearly for a fixed number of warmup steps and decreases proportionally to the inverse square root of the step count thereafter, lr = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5), with warmup_steps set to 1000. Gradients are clipped to a maximum norm of 1.0, and training uses a batch size of 8 sequences throughout. All experiments reported in this paper are run on a single Apple Silicon GPU using PyTorch's MPS backend.

### 4.2 Dataset Splits and Scale

We use the official MAESTRO v3.0.0 train, validation, and test splits, containing 962, 137, and 177 pieces respectively. For training and validation, we cap the pool of overlapping 512-token chunks described in Section 3.1 at 1000 training chunks and 200 validation chunks. This cap is applied after tokenizing every matching piece in a split, so it does not change which pieces contribute to training, only how many windows drawn from them are retained, and it keeps each training run computationally tractable while still drawing chunks from the full breadth of the training and validation pieces. For the held-out generalization evaluation reported in Section 5, we do not apply this cap: the full test split is tokenized and windowed without truncation, yielding 11,451 chunks from all 177 test pieces.

### 4.3 Repeated-Seed Protocol

Every core comparison in this paper is repeated across three random seeds, 0, 1, and 2. Each seed controls the initial weights of the model, the dropout pattern applied during training, and the order in which training chunks are shuffled each epoch. We verified that training two models with the same seed produces identical initial weights and that different seeds produce different initial weights before relying on this protocol. Because the same three seed values are used across every configuration, and because chunk selection is deterministic given the caps in Section 4.2, differences between seeds within a configuration isolate the effect of initialization, dropout, and batch order alone, not differences in the underlying data.

### 4.4 Experimental Design

Table 3 summarizes the experiments reported in this paper: the attention mechanisms and loss functions each covers, the number of seeds and training epochs used, and the research question in Section 1 each addresses. Two scoping decisions are worth stating explicitly. First, cosine attention is retained through 50 epochs and the rogue-dimension diagnostic, but is not extended to the 100-epoch, three-seed protocol used for Gaussian and CosFormer attention, since it is already established as the weakest of the three mechanisms at 50 epochs (Section 5.1) and re-testing it at greater length and cost was not judged a priority given the available compute budget. Second, the rogue-dimension diagnostic in Section 5.5 uses a single-seed matrix of checkpoints trained before the repeated-seed protocol in Section 4.3 was introduced into our experimental pipeline; we treat its findings as descriptive of representational structure rather than as a claim requiring seed-level statistical comparison, since the diagnostic characterizes a trained model's internal geometry rather than a training outcome that varies meaningfully with initialization.

Table 3: Experimental design overview.

| Experiment | Attention types | Loss types | Seeds | Epochs | Research question |
|---|---|---|---|---|---|
| Attention-mechanism comparison, 50 epochs | Gaussian, Cosine, CosFormer | CE | 3 (0, 1, 2) | 50 | RQ1 |
| Attention-mechanism comparison, 100 epochs | Gaussian, Cosine, CosFormer | CE | 3 for Gaussian and CosFormer, 1 (seed 0) for Cosine | 100 | RQ1 |
| Q/K standardization ablation | Gaussian, Cosine | CE, with and without standardization | 3 (0, 1, 2) | 50 | RQ3 |
| Rogue-dimension diagnostic | Gaussian, Cosine, CosFormer | CE, MSE, CE+NTL | 1 (single-seed matrix) | 50 | RQ3 |
| Loss-function comparison | Gaussian, CosFormer | CE, CE+NTL | 3 (0, 1, 2) | 100 | RQ2 |
| Held-out test-set evaluation | Gaussian, Cosine, CosFormer | CE | 1 (seed 0 checkpoints) | 100 | RQ1 |
| Ordinal-token closeness evaluation | Gaussian, CosFormer | CE, CE+NTL | 3 (0, 1, 2) | 100 | RQ2 |

### 4.5 Evaluation Metrics and Protocol

Our primary metric throughout is validation cross-entropy loss, computed on the held-out validation chunks described in Section 4.2 at the end of every training epoch. For each run, we report the loss of the checkpoint with the lowest validation loss, which in every run reported in this paper coincided with the final epoch, since validation loss decreased monotonically throughout training in every configuration we ran. For repeated-seed comparisons, we report the mean and standard deviation of this final validation loss across the three seeds in a configuration, together with the paired difference between two configurations on the same seed, which is more informative than the difference of means alone when seed-to-seed variance is large relative to the effect being measured. We also report perplexity, exp(loss), the effective number of plausible next tokens under the model, alongside raw loss where useful for interpretation.

For the loss-function comparison, comparing a checkpoint's own training loss directly against a checkpoint trained with a different loss is not meaningful when the two losses are not on the same scale, which is the case for cross-entropy and cross-entropy with Number Token Loss, since the latter includes an additive regression term the former does not. We therefore additionally re-evaluate every checkpoint from the loss-function comparison under two fixed metrics, computed identically regardless of which loss the checkpoint was actually trained with: plain cross-entropy, to isolate next-token prediction quality, and the ordinal-token mean squared error described in Section 3.5, to isolate the numerical-closeness objective Number Token Loss is designed to improve. Both re-evaluations use the same 200-chunk validation set described in Section 4.2.

For the rogue-dimension diagnostic, we report the top-1, top-3, and top-5 dimensional concentration ratios described in Section 3.6 as a multiple of the uniform baseline k/d, which allows concentration to be compared across representations of different dimensionality on a common scale.

---

## 5. Results

### 5.1 Attention-Mechanism Comparison

At 50 epochs, averaged over three seeds, Gaussian, Cosine, and CosFormer attention reach mean final validation losses of 5.158 (standard deviation 0.089), 5.282 (standard deviation 0.030), and 5.149 (standard deviation 0.091) respectively (Table 4). CosFormer attention has the lowest mean, but its standard deviation is large enough that it overlaps Gaussian's mean well within one standard deviation, so the two are not distinguishable at this sample size. Cosine attention, by contrast, is separated from the other two by a margin of roughly 0.13, several times its own standard deviation of 0.030, and this separation is robust: cosine attention is the weakest of the three mechanisms throughout every experiment in this paper.

Table 4: Attention-mechanism comparison at 50 epochs, mean and standard deviation over three seeds, cross-entropy loss.

| Attention | Mean validation loss | Std dev |
|---|---|---|
| Gaussian | 5.158 | 0.089 |
| Cosine | 5.282 | 0.030 |
| CosFormer | 5.149 | 0.091 |

[Figure 1: insert mps_experiments/repeated_seeds/plots/mean_std_bar_comparison.png here]

None of the three models had converged at 50 epochs: validation loss was still decreasing at the final epoch in every run. We therefore extended training to 100 epochs and, since the tie between Gaussian and CosFormer could not be resolved by a single additional training length, added two further seeds (1 and 2) to both mechanisms at 100 epochs, reusing the existing seed-0 checkpoints. Cosine attention was not extended to this protocol, since its 50-epoch result already places it clearly behind the other two.

At 100 epochs, Gaussian and CosFormer again reach statistically indistinguishable mean validation losses, 4.9701 (standard deviation 0.0743) and 4.9681 (standard deviation 0.0755) respectively, a difference of 0.0020, about three percent of one standard deviation (Table 5). The paired, same-seed comparison in Table 6 shows no consistent winner: CosFormer is ahead on two of the three seeds and Gaussian on one, with all three margins small. This is the signature of two mechanisms with genuinely equivalent performance at this model and data scale, rather than a real difference obscured by noise, which would instead show a consistent direction across seeds even if the magnitude were small.

Table 5: Attention-mechanism comparison at 100 epochs, mean and standard deviation over three seeds, cross-entropy loss.

| Attention | Mean validation loss | Std dev |
|---|---|---|
| Gaussian | 4.9701 | 0.0743 |
| CosFormer | 4.9681 | 0.0755 |

Table 6: Paired, same-seed comparison at 100 epochs.

| Seed | Gaussian | CosFormer | CosFormer minus Gaussian |
|---|---|---|---|
| 0 | 5.0361 | 5.0254 | -0.0107 |
| 1 | 4.9847 | 4.9963 | +0.0116 |
| 2 | 4.8896 | 4.8826 | -0.0070 |

[Figure 2: insert mps_experiments_100ep/plots/tie_resolution_mean_std.png here]

Every one of the six 100-epoch runs summarized in Table 5 was still decreasing at the final epoch, so these numbers describe where training stood after 100 epochs on this data budget, not a converged ceiling for either mechanism.

Training speed differs substantially between the three mechanisms and is a more decisive distinguishing factor than validation loss. Averaged over the 100-epoch runs above (Table 7), CosFormer trains fastest at approximately 11.0 minutes, Cosine is close behind at approximately 26.5 minutes, measured once, and Gaussian is the slowest at approximately 31.1 minutes, excluding one run that took 77.9 minutes due to a documented host-machine scheduling interruption unrelated to training itself; the other two Gaussian seeds took 25.6 and 36.6 minutes, consistent with the reported average. This ordering is consistent with Gaussian attention's exact, quadratic-complexity kernel (Section 3.3), while Cosine and CosFormer both use a linear-complexity, cumulative-sum formulation.

Table 7: Mean training time, 100 epochs.

| Attention | Mean training time (minutes) |
|---|---|
| CosFormer | 11.0 |
| Cosine | 26.5 (single measurement) |
| Gaussian | 31.1 (excludes one anomalous run, see text) |

### 5.2 Held-Out Test-Set Evaluation

To test whether the comparisons in Section 5.1 reflect genuine generalization rather than overfitting to the 200-chunk validation set used during training, we evaluate the seed-0, 100-epoch checkpoint for each attention type on the full, uncapped MAESTRO test split of 11,451 chunks, more than fifty times the size of the validation set and drawn from 177 pieces never used in training or validation. Table 8 reports both losses.

Table 8: Validation loss versus test loss, 100-epoch seed-0 checkpoints.

| Attention | Validation loss | Test loss | Gap |
|---|---|---|---|
| Gaussian | 5.0361 | 5.0457 | 0.0096 |
| Cosine | 5.0917 | 5.0986 | 0.0069 |
| CosFormer | 5.0254 | 5.0347 | 0.0093 |

[Figure 3: insert mps_experiments/test_evaluation/plots/val_vs_test_comparison.png here]

All three gaps are small and similar in size, roughly 0.007 to 0.010, an order of magnitude smaller than the seed-to-seed standard deviation reported in Section 5.1. We take this as evidence against overfitting to the validation set at this model and data scale. The attention-type ranking established in Section 5.1 also transfers exactly to the test set: CosFormer has the lowest test loss, Gaussian is close behind, and Cosine remains clearly the worst of the three, with a gap to the other two of comparable size to its validation-set gap. Because this evaluation uses only the seed-0 checkpoints, it corroborates the ranking on a larger and disjoint sample of data rather than providing new evidence toward resolving the Gaussian-CosFormer tie itself, which requires seed diversity rather than data diversity.

### 5.3 Loss-Function Comparison

Table 9 compares each checkpoint's own training loss for cross-entropy and cross-entropy with Number Token Loss. At face value, CE+NTL looks worse for both attention types by a nearly identical margin, 0.0349, and the paired per-seed gap is tightly clustered between 0.0343 and 0.0359 across all six runs.

Table 9: Raw logged validation loss, each checkpoint's own training objective, mean over three seeds.

| Attention | CE | CE+NTL | Difference |
|---|---|---|---|
| Gaussian | 4.9701 | 5.0050 | +0.0349 |
| CosFormer | 4.9681 | 5.0030 | +0.0349 |

This level of consistency, a near-identical gap across two different attention mechanisms and six independent training runs, is itself informative: genuine differences between conditions elsewhere in this paper vary unpredictably by seed and mechanism (Section 5.1), while an artifact of measurement produces exactly this kind of uniform offset. The explanation is that the CE+NTL objective is CE plus an additive regression term, lambda times L_NTL (Section 3.5), so a CE+NTL checkpoint's logged loss is not on the same scale as a CE checkpoint's logged loss to begin with; the two never measured the same quantity.

To isolate whether NTL training changes next-token prediction quality itself, we re-evaluate all twelve checkpoints, both attention types, both loss functions, all three seeds, under a single fixed metric, plain cross-entropy, regardless of which loss each checkpoint was actually trained with (Section 4.5). Table 10 reports the result.

Table 10: Pure cross-entropy, re-evaluated under a fixed metric regardless of training objective, mean over three seeds.

| Attention | CE-trained | CE+NTL-trained | Difference |
|---|---|---|---|
| Gaussian | 4.9701 | 4.9700 | -0.0001 |
| CosFormer | 4.9681 | 4.9680 | -0.0001 |

[Figure 4: insert mps_experiments_100ep/plots/ntl_vs_ce_pure_ce_comparison.png here]

Under this fixed metric, the gap disappears almost entirely: the difference between CE-trained and CE+NTL-trained models is -0.0001 for both attention types, three orders of magnitude smaller than the seed-to-seed standard deviation of about 0.075. Number Token Loss does not measurably change next-token prediction quality at this model and data scale, for either attention type; the apparent difference in Table 9 is fully attributable to the additive regression term rather than to any change in the underlying model.

### 5.4 Ordinal-Token Closeness

The comparison in Section 5.3 addresses whether NTL changes classification quality, but NTL's actual design goal is different: improving the numerical closeness of the model's ordinal-token predictions, that is, whether predicting a pitch one semitone from the target is treated as a smaller error than predicting a pitch far away, something plain cross-entropy cannot express (Section 3.5). We test this directly by re-evaluating the same twelve checkpoints on ordinal-token mean squared error, again computed identically regardless of training objective. Table 11 reports the result.

Table 11: Ordinal-token mean squared error, re-evaluated under a fixed metric, mean over three seeds.

| Attention | CE-trained | CE+NTL-trained | Difference | Percent change |
|---|---|---|---|---|
| Gaussian | 0.070014 | 0.069900 | -0.000114 | -0.16% |
| CosFormer | 0.069945 | 0.069836 | -0.000109 | -0.16% |

[Figure 5: insert mps_experiments/ordinal_mse_evaluation/plots/ordinal_mse_comparison.png here]

Both differences are small relative to their respective standard deviations, approximately 0.0015 for Gaussian and 0.0017 for CosFormer, corresponding to roughly six to eight percent of one standard deviation. Combined with Table 10, this gives a complete picture across both metrics that matter for evaluating NTL: at this model and data scale, Number Token Loss changes neither next-token classification quality nor the ordinal numerical-closeness objective it is specifically designed to improve, for either attention type tested.

### 5.5 Rogue-Dimension Diagnostic

We apply the diagnostic described in Section 3.6 to the nine single-seed, 50-epoch checkpoints spanning all three attention types and all three loss functions (Section 4.4). Table 12 summarizes the average top-3 concentration ratio, relative to the uniform baseline of 3/d, across all nine models for each representation type.

Table 12: Average top-3 dimensional concentration, relative to uniform baseline, across nine 50-epoch checkpoints.

| Representation | Variance ratio | Cosine-contribution ratio | Mean cosine similarity |
|---|---|---|---|
| Token embedding | 1.13x | 1.18x | 0.000 |
| Hidden state, layer 0 | 2.43x | 3.93x | 0.502 |
| Hidden state, layer 1 | 2.41x | 4.12x | 0.619 |
| Hidden state, final norm | 2.30x | 3.97x | 0.602 |
| Query projection | 1.86x | 3.14x | 0.387 |
| Key projection | 1.84x | 3.04x | 0.386 |

[Figure 6: insert mps_experiments/diagnostics/rogue_dimensions/plots/variance_concentration_heatmap.png here]
[Figure 7: insert mps_experiments/diagnostics/rogue_dimensions/plots/cosine_concentration_heatmap.png here]

Token embeddings show almost no concentration, close to the uniform baseline, and are close to isotropic, with a mean cosine similarity between random embedding pairs of approximately zero. This closely resembles the static, non-contextual embedding baselines, Word2Vec and GloVe, that Timkey and van Schijndel (2021) find are not affected by rogue dimensions, rather than the large pretrained contextual models their diagnostic was designed to characterize. Hidden states show real anisotropy, a mean cosine similarity between random token pairs of 0.50 to 0.62, but this anisotropy is not concentrated in a small number of dimensions: the top-3 variance share is only 2.3 to 2.4 times the uniform baseline, spread across roughly 15 to 25 of the 64 dimensions rather than 1 to 3. Query and key projections, the space in which Gaussian and cosine attention actually compute their similarity scores, show the mildest concentration of any representation, 1.8 to 1.9 times uniform on variance and 3.0 to 3.1 times uniform on cosine contribution.

The single most concentrated cell found anywhere in this diagnostic, out of nine models times eight representations times three metrics, is the query projection at layer 1 of the CosFormer model trained with CE+NTL, evaluated in its 16-dimensional per-head subspace, where the top dimension carries 30 percent of the average cosine-similarity signal against a 6.25 percent uniform baseline, a factor of 4.8. This is the largest concentration observed in this project and is still far below the departures from uniform that Timkey and van Schijndel (2021) report for large pretrained text models, where a single dimension can account for more than 99 percent of cosine similarity, a factor of roughly 700 relative to uniform for their 768-dimensional representations. No systematic difference in concentration was found between the three attention types.

### 5.6 Q/K Standardization Ablation

Section 5.5 shows that Gaussian and cosine attention's query and key representations exhibit mild but real dimensional concentration. We test directly whether correcting this with the standardization step described in Section 3.4 changes performance, using the same repeated-seed protocol as Section 5.1. Table 13 reports the result, alongside the corresponding unstandardized baseline from Table 4.

Table 13: Effect of Q/K standardization, mean and standard deviation over three seeds, 50 epochs, cross-entropy loss.

| Attention | Without standardization | With standardization | Mean difference |
|---|---|---|---|
| Gaussian | 5.158 (std 0.089) | 5.156 (std 0.088) | -0.0018 |
| Cosine | 5.282 (std 0.030) | 5.285 (std 0.031) | +0.0024 |

See Figure 1 for the corresponding plot, which includes both standardized and unstandardized bars for Gaussian and Cosine attention alongside the CosFormer baseline.

Both mean effects are more than an order of magnitude smaller than the seed-to-seed standard deviation for either attention type. The paired, same-seed comparison is more informative than the means alone: for Gaussian attention, standardization improves validation loss in all three seeds, a consistent direction, though the magnitude, 0.0007 to 0.0034, is small. For cosine attention, standardization improves validation loss in one of three seeds and worsens it in the other two, no consistent direction at all. Q/K standardization does not meaningfully improve either attention type at this model and data scale: Gaussian shows a small, directionally consistent but practically negligible benefit, and cosine shows no reliable effect. This is consistent with the mild concentration found in Section 5.5: a correction for a small measured effect produced a correspondingly small result, and cosine attention's existing L2 normalization already partially compensates for dimension-scale imbalance even without explicit standardization.
