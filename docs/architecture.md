# Architectural Specifications & Mathematical Formulation

This document details the neural network architectures, layer configurations, and loss formulations implemented in the StethoSpeech research repository.

---

## 1. Problem Formulation

The Non-Audible Murmur (NAM) translation task maps acoustic vibration signals captured by an electronic stethoscope into audible speech representations. Following the experimental protocol established by Shah et al. (ACM IMWUT 2024), sensing utilizes a 3M Littmann 3200 Electronic Stethoscope secured primarily over the retroauricular mastoid bone (behind the ear) with an elastic headband, complemented by submental (under-chin) and anterolateral neck skin surface contacts. This captures bone-conducted and soft-tissue articulatory vibrations while physically isolating the microphone from airborne acoustic radiation.

Let $X = (x_1, x_2, \dots, x_{T_{\text{nam}}})$ denote the input NAM feature sequence, where $x_t \in \mathbb{R}^{768}$ represents a frame-level HuBERT embedding at 50 Hz.  
Let $Y = (y_1, y_2, \dots, y_{T_{\text{speech}}})$ denote the target speech feature sequence, where $y_t \in \mathbb{R}^{768}$.

The objective is to train a model $f_\theta$ parameterized by $\theta$ such that $\hat{Y} = f_\theta(X)$ minimizes the masked absolute error relative to $Y$.

---

## 2. Phase 4A: Transformer Baseline Architecture

The baseline is a 6-layer Transformer Encoder mapping directly between HuBERT representation spaces:

$$\hat{Y} = \text{Linear}_{512 \to 768}\left(\text{TransformerEncoder}\left(\text{Linear}_{768 \to 512}(X) + \text{PE}(X)\right)\right)$$

### 2.1 Hyperparameters & Dimensions
- Input Feature Dimension: 768
- Internal Model Dimension ($d_{\text{model}}$): 512
- Feed-Forward Inner Dimension ($d_{\text{ff}}$): 2048
- Attention Heads ($n_{\text{head}}$): 8 ($d_{\text{head}} = 64$)
- Number of Encoder Layers: 6
- Dropout Rate: 0.1
- **Exact Parameter Count:** 19,702,016 parameters

### 2.2 Mathematical Breakdown of Parameters
1. **Input Projection:** $\text{Linear}(768, 512) \implies (768 \times 512) + 512 = 393,728$
2. **Output Projection:** $\text{Linear}(512, 768) \implies (512 \times 768) + 768 = 393,984$
3. **Per Transformer Encoder Layer:**
   - Multi-Head Self-Attention:
     - Input projection: $(512 \times 1536) + 1536 = 787,968$
     - Output projection: $(512 \times 512) + 512 = 262,656$
     - LayerNorm 1: $512 \times 2 = 1,024$
   - Feed-Forward Network:
     - Linear 1: $(512 \times 2048) + 2048 = 1,050,624$
     - Linear 2: $(2048 \times 512) + 512 = 1,049,088$
     - LayerNorm 2: $512 \times 2 = 1,024$
   - Total per layer: $787,968 + 262,656 + 1,024 + 1,050,624 + 1,049,088 + 1,024 = 3,152,384$
4. **6 Encoder Layers:** $6 \times 3,152,384 = 18,914,304$
5. **Grand Total:** $393,728 + 393,984 + 18,914,304 = \mathbf{19,702,016}$

---

## 3. Phase 4B: StethoSpeech Feed-Forward Transformer (FFT)

Adapted from the official StethoSpeech architecture (Shah et al., ACM IMWUT 2024), this model uses 1D convolutional layers within each FFT block to expand temporal receptive fields.

### 3.1 FFT Block Formulation
Each FFT block consists of multi-head self-attention followed by two 1D convolutional layers:

$$\text{FFTBlock}(H) = \text{LayerNorm}\left(H' + \text{Dropout}\left(\text{Conv1D}_{k_2}\left(\text{ReLU}\left(\text{Conv1D}_{k_1}\left(H'\right)\right)\right)\right)\right)$$

where $H' = \text{LayerNorm}(H + \text{Dropout}(\text{MultiHeadAttention}(H)))$.

- Convolution 1 ($k_1 = 9$): Expands receptive field across 180ms of surrounding context (9 frames at 50 Hz). Channels: $256 \to 1024$.
- Convolution 2 ($k_2 = 1$): Projects back to model dimension. Channels: $1024 \to 256$.

### 3.2 Model Parameters Breakdown
- Input Projection: $\text{Linear}(768, 256) \implies 196,864$
- Output Projection: $\text{Linear}(256, 768) \implies 197,376$
- Per FFT Block ($d_{\text{model}}=256, n_{\text{head}}=2, d_{\text{inner}}=1024$):
  - Attention: $(256 \times 768 + 768) + (256 \times 256 + 256) = 263,168$
  - Norm 1: $256 \times 2 = 512$
  - Conv 1: $(256 \times 1024 \times 9) + 1024 = 2,360,320$
  - Conv 2: $(1024 \times 256 \times 1) + 256 = 262,400$
  - Norm 2: $256 \times 2 = 512$
  - Total per FFT block: $2,886,912$
- 6 Encoder Blocks: $6 \times 2,886,912 = 17,321,472$
- 4 Decoder Blocks: $4 \times 2,886,912 = 11,547,648$
- **Total Base Model:** $196,864 + 197,376 + 17,321,472 + 11,547,648 = \mathbf{29,263,360}$

### 3.3 Auxiliary CTC Head Status
- Projection Layers: $\text{Linear}(256, 768) \implies 197,376$ + $\text{Linear}(768, 32) \implies 24,608 = 221,984$ weights.
- Total with CTC: $29,263,360 + 221,984 = \mathbf{29,485,344}$.
- **Empirical Status:** The linear projection head is defined in the network AST for structural fidelity to the paper, but was **decoupled and unused** in the executed training objective.

---

## 4. Phase 5: Unit HiFi-GAN Vocoder (Pure PyTorch)

- Unit Embedding: $1000 \to 128$ dimensions.
- Pre-convolution: $\text{Conv1D}(128, 512, k=7)$.
- Transposed Convolutions: 5 stages with upsampling rates $[5, 4, 4, 2, 2]$.
  $$\text{Total Upsampling Factor} = 5 \times 4 \times 4 \times 2 \times 2 = \mathbf{320\times}$$
  $50 \text{ Hz frames} \times 320 = 16,000 \text{ Hz audio waveform samples}$.
- Multi-Receptive Field (MRF) fusion modules across kernel sizes $[3, 7, 11]$ and dilations $[(1, 3, 5)]$.
- **Exact Parameter Count:** 13,375,809 parameters.

---

## 5. Model Interface Compatibility & Padding Masks

The repository unifies two distinct neural architectures under a common training loop via `ModelTrainer.forward_model()`:
- **Transformer Baseline:** PyTorch's native `nn.TransformerEncoder` expects `src_key_padding_mask` where `True` indicates padded/ignored positions.
- **StethoSpeech FFT Architecture:** Custom FFT layers accept `mask` (or `src_key_padding_mask`), generating inverted 1/0 attention weights and convolutional feature masking.
- **CTC Multi-Output Handling:** For architectures incorporating auxiliary classification heads (e.g. `StethoSpeechWithCTC`), `forward_model()` unpacks the prediction tuple `(pred_features, ctc_logits)` to preserve clean loss computation while maintaining modular decoupling.

---

## 6. Checkpoint Validation & Weight-Norm Reparameterization

To prevent silent fallback to partially initialized random weights and ensure safe execution:
1. **Secure Loading by Default:** All checkpoint loading operations enforce PyTorch's `weights_only=True` to eliminate arbitrary code execution risks.
2. **Strict Parameter Validation:** Checkpoint loaders strictly verify model and vocoder state dictionaries. Any missing parameters (`KeyError`), unexpected keys (`KeyError`), or tensor shape mismatches (`ValueError`) trigger an immediate halt.
3. **Weight Normalization Reparameterization:** Official Meta/fairseq Unit HiFi-GAN checkpoints decouple convolutional filters into magnitude $g$ (`weight_g`) and direction $v$ (`weight_v`). The repository provides automated reparameterization:
   $$W = g \cdot \frac{v}{\|v\|}$$
   evaluated over spatial and channel dimensions, allowing official checkpoints to be ingested into pure PyTorch convolutional layers under strict validation without requiring fairseq dependencies.

