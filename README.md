# Awesome Multi-Modal Time Series Analysis Papers (MM4TSA)

A curated list of papers in the emerging field of multi-modal time series analysis, where multiple data modalities are combined to enhance time series analysis tasks.

**Sorting Method**: Papers in each table are sorted by publication venue first (published papers take precedence over preprints), then by recency (newer arXiv preprints appear before older ones).

## Table of Contents

- [1. Time2X and X2Time: Cross-Modality Interaction for Advanced TSA](#1-time2x-and-x2time-cross-modality-interaction-for-advanced-tsa)
  - [1.1 Text to Time Series](#11-text-to-time-series) - [Jump to table](#text-to-time-series-papers)
  - [1.2 Time Series to Text](#12-time-series-to-text) - [Jump to table](#time-series-to-text-papers)
  - [1.3 Text to Time + Time to Text](#13-text-to-time--time-to-text) - [Jump to table](#text-to-time--time-to-text-papers)
  - [1.4 Domain Specific Applications](#14-domain-specific-applications)
    - [1.4.1 Spatial-Temporal Data](#141-spatial-temporal-data) - [Jump to table](#spatial-temporal-data-papers-1)
    - [1.4.2 Medical Time Series](#142-medical-time-series) - [Jump to table](#medical-time-series-papers-1)
    - [1.4.3 Financial Time Series](#143-financial-time-series) - [Jump to table](#financial-time-series-papers-1)
  - [1.5 Gaps and Outlooks](#15-gaps-and-outlooks) - [Jump to table](#gaps-and-outlooks-papers-1)

- [2. Time+X: Multimodal Extension for Enhanced TSA](#2-timex-multimodal-extension-for-enhanced-tsa)
  - [2.1 Time Series + Text](#21-time-series--text) - [Jump to table](#time-series--text-papers)
  - [2.2 Time Series + Other Modalities](#22-time-series--other-modalities) - [Jump to table](#time-series--other-modalities-papers)
  - [2.3 Domain Specific Applications](#23-domain-specific-applications)
    - [2.3.1 Spatial-Temporal Data](#231-spatial-temporal-data) - [Jump to table](#spatial-temporal-data-papers-2)
    - [2.3.2 Medical Time Series](#232-medical-time-series) - [Jump to table](#medical-time-series-papers-2)
    - [2.3.3 Financial Time Series](#233-financial-time-series) - [Jump to table](#financial-time-series-papers-2)
  - [2.4 Gaps and Outlooks](#24-gaps-and-outlooks) - [Jump to table](#gaps-and-outlooks-papers-2)

- [3. TimeAsX: Reusing Foundation Models for Efficient TSA](#3-timeasx-reusing-foundation-models-for-efficient-tsa)
  - [3.1 Time Series as Text](#31-time-series-as-text) - [Jump to table](#time-series-as-text-papers)
  - [3.2 Time Series as Image](#32-time-series-as-image) - [Jump to table](#time-series-as-image-papers)
  - [3.3 Time Series as Other Modalities](#33-time-series-as-other-modalities)
    - [3.3.1 Tabular Data](#331-tabular-data) - [Jump to table](#tabular-data-papers)
    - [3.3.2 Audio Data](#332-audio-data) - [Jump to table](#audio-data-papers)
  - [3.4 Domain Specific Applications](#34-domain-specific-applications)
    - [3.4.1 Spatial-Temporal Data](#341-spatial-temporal-data) - [Jump to table](#spatial-temporal-data-papers-3)
    - [3.4.2 Medical Time Series](#342-medical-time-series) - [Jump to table](#medical-time-series-papers-3)
    - [3.4.3 Financial Time Series](#343-financial-time-series) - [Jump to table](#financial-time-series-papers-3)
  - [3.5 Gaps and Outlooks](#35-gaps-and-outlooks) - [Jump to table](#gaps-and-outlooks-papers-3)

## 1. Time2X and X2Time: Cross-Modality Interaction for Advanced TSA

### 1.1 Text to Time Series

<a id="text-to-time-series-papers"></a>

| Title | Venue |
|------|---------|
| Language models are zero-shot time series forecasters | NeurIPS 2024 |
| Lag-llama: Towards foundation models for time series forecasting | arXiv 2310 |
| MinIGPT-4: Enhancing vision-language understanding with advanced large language models | arXiv 2304 |
| ChatTS: A LLM-based Time Series Forecasting and Reasoning System | arXiv 2312 |
| CAI-TS-Competition: Evaluating GPT-4 on Time Series Forecasting | arXiv 2406 |
| CLASP: Classifying time series with self-supervised pretraining | arXiv 2404 |

### 1.2 Time Series to Text

<a id="time-series-to-text-papers"></a>

| Title | Venue |
|------|---------|
| T2T-VisDialog: Generating Explanations for Visual Time Series Forecasting | AAAI 2021 |
| Neural Spatio-Temporal PDE Captions | NeurIPS 2020 |
| Repr2Seq: Leveraging Deep Representations for Time Series Forecasting | ICLR 2023 |
| Insight: Temporal Graph Forecasting with Verbal Explanation | AAAI 2023 |
| Domain Knowledge-Enhanced Time Series Captioning | EMNLP 2024 |
| Decoding Time Series Dynamics with Language Model | arXiv 2405 |
| Time Series Captioning: Understanding Time Series with Words | arXiv 2402 |
| XForecast: Explainable Business Time Series Forecasting | arXiv 2311 |
| Explainable Graph Neural Network for Multi-Step Time Series Forecasting | arXiv 2402 |
| Large Language Models are Zero-shot Anomaly Detectors | arXiv 2305 |

### 1.3 Text to Time + Time to Text

<a id="text-to-time--time-to-text-papers"></a>

| Title | Venue |
|------|---------|
| Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis | NeurIPS 2024 |
| ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data | arXiv 2412 |
| ChatTS: A LLM-based Time Series Forecasting and Reasoning System | arXiv 2312 |
| Multi-Modal Forecaster: Jointly Predicting Time Series and Textual Data | arXiv 2411 |
| DataNarrative: Generating Multi-Modal Narratives from Time Series Data | arXiv 2403 |

### 1.4 Domain Specific Applications

#### 1.4.1 Spatial-Temporal Data

<a id="spatial-temporal-data-papers-1"></a>

| Title | Venue |
|------|---------|
| A review on spatiotemporal data models | Journal of Spatial Information Science 2017 |
| A review on spatiotemporal analysis methods for urban crime studies | International Journal of Geo-Information 2019 |
| TeoChat: Multimodal Large Language Model for Integrated Analysis & Design of Territorial Systems | arXiv 2403 |
| UrbanCLIP: Learning Urban Representation with Contrastive Language-Image-POI Pretraining | arXiv 2406 |
| UrbanGPT: Spatio-Temporal Large Language Models | arXiv 2403 |

#### 1.4.2 Medical Time Series

<a id="medical-time-series-papers-1"></a>

| Title | Venue |
|------|---------|
| Automated Detection and Reporting of Significant ECG Changes Using Generative AI | AAAI 2024 |
| Frozen language model helps ECG zero-shot learning | ICLR 2024 |
| Using grammatical relations to automate the generation of text descriptions from medical time series | MedInfo 2008 |
| Data-to-text generation with attention-based neural networks for the generation of clinical narratives | Knowledge Management & E-Learning 2009 |
| Trends in automatic generation of clinical summaries | ACM 2008 |
| Electrocardiogram reasoning with large language models | Scientific Reports 2023 |
| MEIT: Leveraging Multimodal Clinical Notes to Enhance Time-Series Modeling for Patient Risk Assessment | arXiv 2404 |
| Enhancing Electrocardiogram Analysis with Large Language Models: A Comprehensive Explanation Framework | arXiv 2405 |
| Diffusion for time-series imputation and forecasting | arXiv 2306 |
| Exploring the use of text-to-image models for biomedical image generation | arXiv 2307 |
| DiffusETS: A Diffusion-Based Ensemble of Time Series Forecasting Models for Improved Uncertainty Quantification | arXiv 2401 |
| ECG-LLM: Integrating Expert Knowledge into Large Language Model for ECG Analysis | arXiv 2401 |
| Multimodal Generative Learning on Medical Time Series | arXiv 2404 |
| ECG Text Generation | arXiv 2402 |
| BioSignal-LLM: Decoding Brain and Heart for Multimodal Large Language Model | arXiv 2304 |
| MedTS-LLM: Exploring the Potentials of Large Language Models for Interpretable Medical Time Series Analysis | arXiv 2401 |
| Towards Explainable Multimodal Arrhythmia Classification | arXiv 2403 |

#### 1.4.3 Financial Time Series

<a id="financial-time-series-papers-1"></a>

| Title | Venue |
|------|---------|
| Long memory time series forecasting using BERT | NEUCOM 2022 |
| Neural News Filtering for Accurate Stock Movement Prediction | IEEE 2023 |
| Knowledge-enhanced Multimodal Fusion with Large Language Models for Stock Movement Prediction | AAAI 2024 |
| Multimodal learning for stock movement prediction: A survey | Information Fusion 2023 |
| Diffusion for time-series imputation and forecasting | arXiv 2306 |
| FinTRAL: Financial Report Text Representation and Adversarial Learning for Stock Movement Prediction | arXiv 2401 |
| Open-Source Large Language Model Beats GPT-4 in Financial Sentiment Analysis | arXiv 2401 |

### 1.5 Gaps and Outlooks

<a id="gaps-and-outlooks-papers-1"></a>

| Title | Venue |
|------|---------|
| Evaluating System 1 vs. System 2 Processing of Time Series Forecasting in Large Language Models | arXiv 2401 |
| Beyond Prediction: Programmatically Generating Rationales for Time Series Analysis | arXiv 2407 |
| Picture Worth A Thousand Words: Image Improves In-Context Learning for Time Series Tasks | arXiv 2405 |
| ChatTS: A LLM-based Time Series Forecasting and Reasoning System | arXiv 2312 |

## 2. Time+X: Multimodal Extension for Enhanced TSA

### 2.1 Time Series + Text

<a id="time-series--text-papers"></a>

| Title | Venue |
|------|---------|
| Time-LLM: Time Series Forecasting by Reprogramming Large Language Models | ICLR 2023 |
| Time-Series Forecasting with Transformer and Temporal Guidance | NeurIPS 2023 |
| Chain-of-thought prompting elicits reasoning in large language models | NeurIPS 2022 |
| LSTPrompt: Large Language Models as Zero-Shot Time Series Forecasters by Long-Short-Term Prompting | ACL 2024 |
| Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis | NeurIPS 2024 |
| From news to forecast: Integrating event analysis in llm-based time series forecasting with reflection | NeurIPS 2025 |
| Time-Series Forecasting for Out-of-Distribution Generalization Using Invariant Learning | ICML 2024 |
| Context is Key: A Benchmark for Forecasting with Essential Textual Information | arXiv 2410 |
| Dual-Forecaster: A Multimodal Time Series Model Integrating Descriptive and Predictive Texts | ICLR 2025 |
| ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data | arXiv 2412 |
| AutoTimeS: Automatic Time Series Forecasting with Prompt-Enhanced LLM | arXiv 2401 |
| Beyond trend and periodicity: Guiding time series forecasting with textual cues | arXiv 2405 |
| FNSPID: A Comprehensive Financial News Dataset in Time Series | arXiv 2402 |
| Advancing Time Series Forecasting with Language Models: Context and Retrieval-based Prompting | arXiv 2411 |
| Hierarchical Contextual Prompting for Time Series Forecasting with Large Language Models | arXiv 2406 |
| Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative | arXiv 2502 |
| DualTime: Improving Anomaly Detection in Multimodal Time-Series with Textual Data | arXiv 2404 |

### 2.2 Time Series + Other Modalities

<a id="time-series--other-modalities-papers"></a>

| Title | Venue |
|------|---------|
| ImageBind: One Embedding Space To Bind Them All | CVPR 2023 |
| LANiST-R: Pre-training Multimodal Models for Temporal and Robust Inpatient Mortality Prediction | AMIA 2023 |

### 2.3 Domain Specific Applications

#### 2.3.1 Spatial-Temporal Data

<a id="spatial-temporal-data-papers-2"></a>

| Title | Venue |
|------|---------|
| Urban computing: concepts, methodologies, and applications | ACM TIST 2014 |
| Traffic Data Augmentation via Foundation Models | AAAI 2024 |
| MMST: Multi-Modal Spatial-Temporal Learning for Urban Flow Prediction | ICDE 2023 |
| BJTT: Beijing Jitter Tracking Dataset for Computational Public Transit Research | CIKM 2024 |
| Understanding the role of semantics in mobility data mining | UBIQUITOUS 2016 |
| Spatial-temporal data-driven urban dynamics prediction: A survey | ACM Computing Surveys 2022 |
| Terra: Imperative, Declarative, and Natural Language Interfaces for Geospatial Analysis | arXiv 2404 |
| CityGPT: Generative Pre-trained Transformer for Spatial-Temporal Urban Prediction | arXiv 2404 |
| Event-specific Attention Spatiotemporal DiffusionNet for Transportation Network Modeling | ICDM 2024 |
| Mobile Cellular Traffic Analysis for 6G Networks with Multimodal Deep Learning | arXiv 2407 |

#### 2.3.2 Medical Time Series

<a id="medical-time-series-papers-2"></a>

| Title | Venue |
|------|---------|
| PTB-XL, a large publicly available electrocardiography dataset | Scientific Data 2020 |
| Multimodal fusion of structural MRIs and electroencephalograms for Alzheimer's disease diagnosis using deep learning | Scientific Reports 2020 |
| MIRACLE: Mining complex physiological responses to estimate patient similarity | ACM CHIL 2021 |
| Leveraging multimodal Electronic Health Records to predict heart failure patient readmission | AMIA 2021 |
| Multimodal Representation Learning for ICU Mortality and Readmission Prediction | AAAI 2023 |
| Learning Dual Multi-Scale Vision-Language Representation for Multivariate Time Series Event Classification | MICCAI 2023 |
| Deep multimodal learning for ICU prediction tasks using time series and clinical notes | arXiv 2107 |
| MedFuse: Multimodal Fusion with Clinical Time-Series Data and Chest X-Ray Images | arXiv 2207 |
| Improving clinical outcome prediction by leveraging multimodal feature representations and temporal knowledge | arXiv 2208 |
| Multimodal Embedding Alignment for Effective and Robust Inpatient Mortality Prediction | arXiv 2404 |
| EMERGE: A benchmark for ECG signals with clinical notes | arXiv 2403 |
| Addressing Clinical Reality: The Prediction of In-hospital Mortality and Adverse Events with Multimodal Neural Networks | arXiv 2404 |
| Arbitrary Modality Contrastive Learning and Reasoning for Medical Vision-Language Foundation Model | arXiv 2403 |
| Integrated Human-AI Healthcare Systems with Foundation Models: Vision and Research Opportunities | arXiv 2404 |
| Towards Explainable Multimodal Arrhythmia Classification | arXiv 2403 |

#### 2.3.3 Financial Time Series

<a id="financial-time-series-papers-2"></a>

| Title | Venue |
|------|---------|
| Financial time series forecasting - A deep learning approach | IJCNN 2009 |
| The volatility of narratives: Multidimensional aspects of uncertainty in English language news channels | Information Fusion 2017 |
| Natural language based financial forecasting: a survey | Journal of Artificial Intelligence 2018 |
| Stock Movement Prediction with Financial News using Fine-tuned Transformer Models | IJCNN 2023 |
| Modality-specific and modality-invariant visual and language representations as a unified cross-modal representation for decision making | ICDM 2023 |
| Text2TimeSeries: Learning Time Series Representations with Semantic Guidance for Few-shot Time Series Forecasting | CVPR 2024 |
| Predicting the Stock Market with News Articles | arXiv 2304 |
| Natural Language Processing and Machine Learning for Mortgage Default Prediction: A Systematic Literature Review | arXiv 2308 |
| Multi-modal Deep Learning for Stock Selection with Multi-scale Trading Signal Fusion | arXiv 2311 |
| FNSPID: A Comprehensive Financial News Dataset in Time Series | arXiv 2402 |
| Multi-model Combination for Capturing Conditional Dynamics of Multimodal Financial Time Series | arXiv 2405 |
| Multimodal Machine Learning for Stock Movement Prediction | arXiv 2408 |

### 2.4 Gaps and Outlooks

<a id="gaps-and-outlooks-papers-2"></a>

| Title | Venue |
|------|---------|
| ImageBind: One Embedding Space To Bind Them All | CVPR 2023 |
| LANiST-R: Pre-training Multimodal Models for Temporal and Robust Inpatient Mortality Prediction | AMIA 2023 |

## 3. TimeAsX: Reusing Foundation Models for Efficient TSA

### 3.1 Time Series as Text

<a id="time-series-as-text-papers"></a>

| Title | Venue |
|------|---------|
| Large language models are zero-shot time series forecasters | NeurIPS 2024 |
| One fits all: Power general time series analysis by pretrained lm | NeurIPS 2023 |
| Time-LLM: Time Series Forecasting by Reprogramming Large Language Models | ICLR 2023 |
| Large Pre-trained Time Series Models for Cross-domain Time Series Analysis Tasks | NeurIPS 2024 |
| LSTPrompt: Large Language Models as Zero-Shot Time Series Forecasters by Long-Short-Term Prompting | ACL 2024 |
| Enhancing Large Language Models as Zero-shot Time Series Anomaly Detectors | arXiv 2311 |
| Large Language Models are Zero-shot Anomaly Detectors | arXiv 2305 |
| Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence? | arXiv 2303 |
| Leveraging Chain-of-Thought Prompting and Large Language Models for Enhanced Time Series Analysis | arXiv 2309 |
| Lag-llama: Towards foundation models for time series forecasting | arXiv 2310 |
| Tempo: Prompt-based generative pre-trained transformer for time series forecasting | arXiv 2310 |
| ChatTS: A LLM-based Time Series Forecasting and Reasoning System | arXiv 2312 |
| TEST: Text Prototype Aligned Embedding to Activate LLM Knowledge for Time Series | arXiv 2307 |
| AutoTimeS: Automatic Time Series Forecasting with Prompt-Enhanced LLM | arXiv 2401 |
| Context is Key: A Benchmark for Forecasting with Essential Textual Information | arXiv 2410 |
| S-PromptCast: Semantic Prompting for Time Series Forecasting | arXiv 2404 |
| Multi-Patch Prediction for Time Series Foundation Models | arXiv 2404 |
| Chronos: Learning the language of time series | arXiv 2403 |
| ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data | arXiv 2412 |
| Exploiting In-Batch Semantic Connections for Time Series Forecasting with Large Language Models | arXiv 2405 |

### 3.2 Time Series as Image

<a id="time-series-as-image-papers"></a>

| Title | Venue |
|------|---------|
| An image is worth 16x16 words: Transformers for image recognition at scale | ICLR 2021 |
| Deep convolutional neural networks for image classification: A comprehensive review | Neural computation 2021 |
| Forecasting commodity price time series using long short-term memory neural networks with vision transformers | Expert Systems with Applications 2020 |
| Time Series as Images: Vision Transformer for Irregularly Sampled Time Series | ICLR 2023 |
| Imaging time series to improve classification and imputation | arXiv 1506 |
| ViTime: Vision Transformer for Multivariate Time Series Forecasting | arXiv 2401 |
| PLOTS: An LLM-based Foundation Model for Reasoning and State-tracking in Time-series | arXiv 2401 |
| ViTime: Enhance Pretrained Vision Transformers for Time Series Forecasting | arXiv 2407 |
| See It, Think It, Sorted: Large Multimodal Models are Few-shot Time Series Anomaly Analyzers | arXiv 2411 |
| VisionTS: Foundation Models as Zero-shot Time Series Predictors and Anomaly Detectors | arXiv 2405 |
| Can Vision Perceive Time? Vision Foundation Models for Time Series Forecasting | arXiv 2406 |
| Training Neural Networks with Tiny Synthetic Datasets | arXiv 2405 |
| Towards a comprehensive survey of vision-based time series analysis | arXiv 2212 |
| CAFO: Learning Cross-modal Alignment for Multivariate Time Series Forecasting | arXiv 2405 |
| TimeVLM: Multimodal prompting unlocks time series forecasting for vision-language models | arXiv 2405 |

### 3.3 Time Series as Other Modalities

#### 3.3.1 Tabular Data

<a id="tabular-data-papers"></a>

| Title | Venue |
|------|---------|
| Tabular Transformers for Modeling Multivariate Time Series | ICASSP 2020 |
| ForecastPFN: Zero-shot forecasting of time series from a single example | arXiv 2302 |
| Tabular Language Models for Chain of Thought Time Series Reasoning | arXiv 2403 |
| TableTime: Time Series Foundation Model | arXiv 2403 |

#### 3.3.2 Audio Data

<a id="audio-data-papers"></a>

| Title | Venue |
|------|---------|
| Voice2Series: Reprogramming acoustic models for time series classification | ICML 2021 |
| SSAST: Self-Supervised Audio Spectrogram Transformer | AAAI 2022 |
| T-WaveNet: Tree-structured wavelet neural network for partial discharge pulse pattern recognition | IEEE 2021 |

### 3.4 Domain Specific Applications

#### 3.4.1 Spatial-Temporal Data

<a id="spatial-temporal-data-papers-3"></a>

| Title | Venue |
|------|---------|
| Deep spatiotemporal residual networks for citywide crowd flows prediction | AAAI 2017 |
| Deep learning for spatiotemporal predictions | BDAI 2018 |
| STORM: Spatio-temporal online reasoning and learning for multiple moving cameras | CVPR 2022 |
| Learning from irregularly-sampled temporal data with multi-view attention and temporal point processes | ICLR 2023 |
| TrafficGPT: Integrating text and time series for multi-horizon traffic prediction | ICLR 2024 |
| Spatial-Temporal Graph Transformer with Frequency-enhanced Gating for Traffic Flow Forecasting | TNNLS 2024 |
| VMRNN: Integrating Vision Mamba and Deep-learning Models for Short-term Traffic Flow Prediction | IEEE IoT Journal 2024 |
| Can Vision Perceive Multivariate Time Series? | arXiv 2403 |
| UniST: Unified Vision-and-Language Spatiotemporal Learning | arXiv 2401 |
| UrbanGPT: Spatio-Temporal Large Language Models | arXiv 2403 |
| ClimateLLM: Data-Centric Foundation Models for Climate-Related Prediction | arXiv 2401 |
| TPLLM: An Explainable Large Language Model for Traffic Prediction | arXiv 2407 |

#### 3.4.2 Medical Time Series

<a id="medical-time-series-papers-3"></a>

| Title | Venue |
|------|---------|
| Large language models for biosignal processing and outcome prediction in the intensive care unit | NPJ Digital Medicine 2023 |
| Multimodal patient representation learning with missing modalities and labels | arXiv 2306 |
| ECG language model: a pre-trained model for ECG signal processing tasks | arXiv 2402 |
| ECG-LLM: Integrating Expert Knowledge into Large Language Model for ECG Analysis | arXiv 2401 |
| DualTime: Improving Anomaly Detection in Multimodal Time-Series with Textual Data | arXiv 2404 |
| MedTS-LLM: Exploring the Potentials of Large Language Models for Interpretable Medical Time Series Analysis | arXiv 2401 |

#### 3.4.3 Financial Time Series

<a id="financial-time-series-papers-3"></a>

| Title | Venue |
|------|---------|
| A Survey on Computational Intelligence Approaches for Predictive Modeling in Prostate Cancer | Expert Systems with Applications 2019 |
| Forecasting commodity price time series using long short-term memory neural networks with vision transformers | Expert Systems with Applications 2020 |
| A comprehensive review of financial time series prediction using adaptive neuro-fuzzy system | Neural Computing and Applications 2022 |
| DeepWave: A Recurrent Neural-Network for Real-Time Acoustic Imaging | CVPR 2024 |
| Research on stock price prediction based on LSTM-CNN model | The Journal of Supercomputing 2024 |
| Quantum Transformers for Financial Time Series Forecasting | arXiv 2401 |
| Stock-gpt: Large language models and business knowledge for stock market predictions | arXiv 2306 |
| Pixels to pips: Predicting financial time series using images generated from open-source technical analysis charting tools | arXiv 2303 |
| An image is worth 16x16 words: Transformers for image recognition at scale | ICLR 2021 |
| Image-based deep learning price prediction: Practical application to crude oil futures | arXiv 2103 |
| Encoding financial data for deep learning applications in trading | arXiv 2009 |
| Imaging time series to improve classification and imputation | arXiv 1506 |
| Multi-Task Reinforcement Learning with Graph Transformer for Personalized Stock Trading | arXiv 2404 |

### 3.5 Gaps and Outlooks

<a id="gaps-and-outlooks-papers-3"></a>

| Title | Venue |
|------|---------|
| Picture Worth A Thousand Words: Image Improves In-Context Learning for Time Series Tasks | arXiv 2405 |
| Vision for All: Investigating the Landscape and Future of Computer Vision Foundation Models | arXiv 2405 |

---

*This list is regularly updated. Contributions via PRs are welcome to add new papers or correct information.*
