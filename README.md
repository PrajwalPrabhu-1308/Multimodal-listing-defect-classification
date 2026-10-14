# Multimodal E-Commerce Listing Defect Classifier

**Automated product listing quality control using computer vision, natural language processing, multimodal learning, and vision-language model benchmarking.**

## Overview

E-commerce marketplaces rely on accurate product images, descriptive titles, and consistent product metadata. Defects such as image–title mismatches, incorrect color descriptions, and missing product attributes can reduce catalog reliability and create poor customer experiences.

This project develops a multimodal deep-learning pipeline to identify potential defects in product listings using both visual and textual information.

The framework combines a ResNet50 image encoder with a DistilBERT text encoder in a shared multimodal classification architecture. It jointly predicts product categories and three listing-defect labels. A zero-shot Qwen2-VL model provides a separate vision-language baseline for comparison.

The project also incorporates Grad-CAM visual explanations and inference-performance analysis to investigate model behavior and deployment trade-offs.

**Project status:** Research prototype using the Amazon Berkeley Objects dataset and synthetically injected defects.

## Objectives

- Build a multimodal classifier that jointly processes product images and listing titles.
- Predict product categories and multiple listing-defect labels.
- Address class imbalance using Asymmetric Loss.
- Compare trained-model inference against a zero-shot vision-language model.
- Visualize image regions associated with defect predictions using Grad-CAM.
- Estimate inference throughput, latency, and computational cost under a simulated high-volume workload.

## Defect Categories

The model predicts three binary defect labels.

| Defect | Description |
|---|---|
| `is_mismatch` | The product image does not correspond to the expected listing category or product description. |
| `wrong_color` | The color indicated by the title conflicts with the image. |
| `missing_attribute` | The listing title lacks an expected product attribute. |

The model also performs multiclass product-category classification.

The current training labels are generated through synthetic defect injection. Consequently, the classifier learns the patterns represented by the injection procedures, which may differ from naturally occurring catalog defects.

## Methodology

### 1. Dataset preparation

The project uses the [Amazon Berkeley Objects (ABO) dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), which provides product images and listing metadata.

The notebook downloads the small-image archive and listing metadata, resolves image identifiers to local file paths, and joins images with their associated listing information.

The current data preparation configuration targets up to 10,000 listing records before synthetic defect generation. The final number of usable records depends on image availability, metadata matching, and filtering.

### 2. Synthetic defect generation

A custom `SyntheticDefectInjector` creates controlled examples of listing defects.

The process introduces:

- Image–title mismatches by substituting images from products in different categories.
- Color inconsistencies by modifying the relationship between image content and the title's color description.
- Missing-attribute defects by removing or omitting selected title information.

Each listing receives binary labels for the three defect categories.

This controlled setup provides labels for supervised training without requiring a manually annotated defect dataset.

**Evaluation caveat:** Synthetic labels measure performance against the implemented defect-generation rules. They do not independently establish detection accuracy on real-world listing defects.

### 3. Multimodal architecture

The classifier uses two pretrained encoders.

**Visual encoder — ResNet50**

- Initialized with ImageNet-pretrained weights.
- Removes the final classification layer to obtain image embeddings.
- Fine-tunes the final portion of the network while freezing most earlier layers.

**Text encoder — DistilBERT**

- Uses `distilbert-base-uncased`.
- Tokenizes product titles.
- Extracts contextual text representations from the transformer encoder.

**Feature fusion**

The image and text representations are projected into a shared embedding space and combined before passing through task-specific prediction heads.

The architecture supports two related objectives:

1. **Category classification:** Multiclass prediction of the product category.
2. **Defect classification:** Independent binary predictions for image–title mismatch, wrong color, and missing attributes.

This multi-task design allows category recognition and defect identification to share information from the same multimodal representation.

### 4. Multi-task optimization

The training objective combines category-classification loss and defect-classification loss.

Category prediction uses cross-entropy loss. Defect prediction uses Asymmetric Loss (ASL), which applies different focusing parameters to positive and negative examples.

ASL is intended to reduce the influence of easy negative examples and improve learning under imbalanced multi-label conditions.

The model is trained using AdamW with a cosine learning-rate schedule and warm-up.

The current configuration uses:

- An 80/20 training–validation split.
- A fixed random seed of 42 for the split and initial randomization.
- Batch size of 32 for the training and validation loaders.
- Five training epochs.
- An initial learning rate of \(2 \times 10^{-4}\).
- A default defect-prediction threshold of 0.5.

The notebook reports validation loss and defect-classification metrics during training.

### 5. Zero-shot Qwen2-VL benchmark

The trained multimodal classifier is compared with a separate vision-language model:

[Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct).

Qwen2-VL receives the product image, listing title, and product type. It is prompted to return JSON containing the three binary defect predictions.

The benchmark evaluates 50 sample listings and records per-listing inference latency.

This comparison investigates the trade-off between a task-specific classifier trained on synthetic labels and a general-purpose vision-language model prompted to perform the same inspection task.

The two approaches differ in architecture, training objectives, and decision mechanisms. Their results should therefore be interpreted as a practical benchmark rather than a controlled comparison of model architectures alone.

### 6. Explainability with Grad-CAM

The notebook implements Grad-CAM using the final convolutional block of the ResNet50 visual encoder.

For a selected defect output, the procedure:

1. Computes the model's defect predictions.
2. Backpropagates the target defect score.
3. Uses activation maps and gradients to calculate channel weights.
4. Produces a spatial heatmap.
5. Overlays the heatmap on the original image.

The visualization is intended to show which image regions contribute to the selected defect score.

Grad-CAM explains the visual branch's activation patterns. It does not fully explain how the text representation, multimodal fusion, and classification heads jointly determine the final prediction. A highlighted region should not be interpreted as proof of a defect or as a causal explanation.

### 7. Inference performance analysis

The notebook compares measured inference latency for the Qwen2-VL benchmark and the trained PyTorch classifier.

It also includes a simulated large-scale inference workload of 500,000 listings, using GPU autocast and batch-based processing.

The performance summary estimates:

- Per-listing inference latency.
- Throughput.
- Projected processing time for a large workload.
- Approximate inference cost under an assumed GPU hourly rate.
- An illustrative Qwen2-VL cost based on a fixed per-listing pricing assumption.

These estimates are scenario-dependent. The Qwen2-VL cost is a pricing emulation, not a measured cloud API bill. The large-scale workload is a projection, not evidence that 500,000 real listings were processed.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Image encoder | ResNet50 |
| Text encoder | DistilBERT |
| Vision-language baseline | Qwen2-VL-2B-Instruct |
| Image processing | Pillow, OpenCV |
| Data processing | NumPy, pandas |
| Evaluation | scikit-learn |
| Explainability | Grad-CAM |
| Transformer utilities | Hugging Face Transformers |
| Learning-rate scheduling | PyTorch and Transformers |
| Execution environment | Jupyter Notebook / Google Colab |

## Repository Structure

The notebook is located directly in the repository root.

```text
multimodal-listing-defect-classifier/
├── README.md
└── multimodal_listing_defect_classifier_abo.ipynb
```

The notebook contains the current end-to-end implementation, including dataset preparation, synthetic label generation, model definition, training, benchmarking, explainability, and inference analysis.

Separate Python modules, automated tests, and a production inference interface have not yet been extracted.

## Getting Started

### Prerequisites

- Python 3.10 or 3.11 recommended.
- Jupyter Notebook or Google Colab.
- Internet access for dataset downloads and pretrained model weights.
- A CUDA-enabled GPU recommended for model training and Qwen2-VL inference.

The Qwen2-VL benchmark is computationally demanding. GPU memory availability and compatible package versions may affect execution.

### Recommended environment: Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `multimodal_listing_defect_classifier_abo.ipynb` from your repository.
3. Select **Runtime → Change runtime type → T4 GPU**, if available.
4. Run the notebook cells in sequence.
5. Review the training metrics, Qwen2-VL benchmark, Grad-CAM visualizations, and inference-performance summary.

The notebook installs several dependencies and downloads the ABO dataset automatically. Review the package installation cell if your Colab environment uses a different PyTorch or Transformers version.

### Local execution

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the core dependencies:

```bash
python -m pip install --upgrade pip
pip install torch torchvision transformers timm
pip install numpy pandas scikit-learn matplotlib opencv-python-headless
pip install jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Open `multimodal_listing_defect_classifier_abo.ipynb` from the repository root and execute the cells sequentially.

For GPU inference with Qwen2-VL, verify compatibility between PyTorch, CUDA, Transformers, and the model's processor requirements before running the benchmark.

## Evaluation Metrics

The notebook uses defect-classification metrics including:

- Macro F1-score.
- Micro F1-score.
- Per-class F1-score.
- Precision and recall.
- Validation loss.
- Per-listing inference latency.
- Projected throughput and processing cost.

Macro F1 gives equal weight to the three defect classes, while micro F1 aggregates predictions across all label instances. Reporting both helps reveal performance differences under class imbalance.

A default probability threshold of 0.5 is used for binary defect predictions. Threshold calibration on a separate validation set may improve the precision–recall trade-off.

The notebook's generated metrics should be recorded from actual executions rather than assumed or hardcoded into the project documentation.

## Limitations

- **Synthetic defects:** Generated examples may not capture the complexity of real catalog-quality problems.
- **Label-generation dependence:** The classifier may learn artifacts of the synthetic injection process.
- **Category coverage:** The selected ABO records may not represent the full diversity of e-commerce products.
- **Validation scope:** A random training–validation split does not establish generalization to unseen sellers, product families, or future catalog distributions.
- **Multimodal attribution:** Grad-CAM explains visual activations but does not fully attribute predictions across both modalities.
- **Benchmark comparability:** The trained classifier and zero-shot Qwen2-VL use different modeling approaches and may have different preprocessing and compute requirements.
- **Latency variability:** Runtime depends on hardware, batch size, model loading, preprocessing, and GPU utilization.
- **Cost assumptions:** Projected costs depend on assumed compute rates and pricing scenarios.
- **Production readiness:** The notebook is an experimental prototype, not a deployed catalog-quality service.

## Future Work

- Evaluate on independently labeled real-world listing defects.
- Introduce seller-level and product-family-level data splits to assess generalization.
- Calibrate decision thresholds for individual defect classes.
- Compare alternative visual encoders and multimodal fusion strategies.
- Add precision–recall curves and per-defect confusion matrices.
- Investigate text and image attribution methods for the complete multimodal architecture.
- Benchmark multiple batch sizes, hardware configurations, and quantization strategies.
- Extract reusable model, data, and evaluation modules.
- Add automated tests, dependency version pinning, and reproducible experiment configurations.
- Develop an inference interface for catalog-quality review.

## Project Status

**Status: Research prototype / experimental implementation.**

This project investigates whether combining product images and listing text can improve automated defect detection relative to a vision-language model prompted for zero-shot inspection.

The framework integrates multi-task classification, synthetic defect generation, model benchmarking, visual explanations, and inference analysis. Real-world validation is required before making claims about operational accuracy or production suitability.

## References

- [Amazon Berkeley Objects dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
- [ResNet paper — Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [DistilBERT paper — A Distilled Version of BERT](https://arxiv.org/abs/1910.01108)
- [Asymmetric Loss for Multi-Label Classification](https://arxiv.org/abs/2009.14119)
- [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/)
- [PyTorch documentation](https://pytorch.org/docs/stable/)
