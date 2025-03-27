<div align="center">
  <h1>Awesome Multi-Modalities For Time Series Analysis Papers (MM4TSA)</h1>
  <p>
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
    <img src="https://img.shields.io/badge/PRs-Welcome-green" alt="PRs Welcome">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://arxiv.org/abs/2503.11835"><img src="https://img.shields.io/badge/arXiv-2503.11835-b31b1b.svg" alt="arXiv"></a>
  </p>
</div>
<div align="center">

**[<a href="https://mp.weixin.qq.com/s/DUgaWPuf0g5EdUlGulADWA">中文解读1</a>]**
**[<a href="https://mp.weixin.qq.com/s/ytGc3c_ZaPpoz2pNsydH3w">中文解读2</a>]**

</div>

---
Time series analysis (TSA) is a longstanding research topic in the data mining community and has wide real-world significance. Compared to "richer" modalities such as language and vision, which have recently experienced explosive development and are densely connected, the time-series modality remains relatively underexplored and isolated. We notice that many recent TSA works have formed a new research field, i.e., Multiple Modalities for TSA (MM4TSA). In general, these MM4TSA works follow a common motivation: how TSA can benefit from multiple modalities. This survey is the first to offer a comprehensive review and a detailed outlook for this emerging field. Specifically, we systematically discuss three benefits: (1) reusing foundation models of other modalities for efficient TSA, (2) multimodal extension for enhanced TSA, and (3) cross-modality interaction for advanced TSA. We further group the works by the introduced modality type, including text, images, audio, tables, and others, within each perspective. Finally, we identify the gaps with future opportunities, including the reused modalities selections, heterogeneous modality combinations, and unseen tasks generalizations, corresponding to the three benefits. We release this up-to-date GitHub repository that includes key papers and resources. More details please check our <a href="https://arxiv.org/abs/2503.11835"><strong>survey</strong></a>.

<div align="center">
    <img src="https://github.com/AdityaLab/MM4TSA/blob/main/Survey_Logo_1.jpg" width="447">
  <img src="https://github.com/AdityaLab/MM4TSA/blob/main/Survey_Logo_2.jpg" width="500">
</div>


## Contributing

🚀 We will continue to update this repo. If you find it helpful, please Star it or Cite Our Survey.

🤝 Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

🙏 🙏 🙏If you find this survey useful, please consider citing our paper.


```
@misc{liu2025timeseriesanalysisbenefit,
      title={How Can Time Series Analysis Benefit From Multiple Modalities? A Survey and Outlook}, 
      author={Haoxin Liu and Harshavardhan Kamarthi and Zhiyuan Zhao and Shangqing Xu and Shiyu Wang and Qingsong Wen and Tom Hartvigsen and Fei Wang and B. Aditya Prakash},
      year={2025},
      eprint={2503.11835},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.11835}, 
}
```


---

## Table of Contents

- [1. Time2X and X2Time](#1-time2x-and-x2time)
  - [1.1 Text to Time Series](#11-text-to-time-series)
    - [1.1.1 Generation](#111-generation)
    - [1.1.2 Retrieval](#112-retrieval)
  - [1.2 Time Series to Text](#12-time-series-to-text)
    - [1.2.1 Explanation](#121-explanation)
    - [1.2.2 Captioning](#122-captioning)
  - [1.3 Text to Time + Time to Text](#13-text-to-time-+-time-to-text)
  - [1.4 Other Cross-Modality Works](#14-other-cross-modality-works)
  - [1.5 Domain Specific Applications](#15-domain-specific-applications)
    - [1.5.1 Spatial-Temporal Data](#151-spatial-temporal-data)
    - [1.5.2 Medical Time Series](#152-medical-time-series)
    - [1.5.3 Financial Time Series](#153-financial-time-series)
  - [1.6 Gaps and Outlooks](#16-gaps-and-outlooks)
- [2. Time+X](#2-time+x)
  - [2.1 Time Series + Text](#21-time-series-+-text)
  - [2.2 Time Series + Other Modalities](#22-time-series-+-other-modalities)
  - [2.3 Domain Specific Applications](#23-domain-specific-applications)
    - [2.3.1 Spatial-Temporal Data](#231-spatial-temporal-data)
    - [2.3.2 Medical Time Series](#232-medical-time-series)
    - [2.3.3 Financial Time Series](#233-financial-time-series)
  - [2.4 Gaps and Outlooks](#24-gaps-and-outlooks)
- [3. TimeAsX](#3-timeasx)
  - [3.1 Time Series as Text](#31-time-series-as-text)
  - [3.2 Time Series as Image](#32-time-series-as-image)
  - [3.3 Time Series as Other Modalities](#3-time-series-as-other-modalities)
    - [3.3.1 Tabular Data](#331-tabular-data)
    - [3.3.2 Audio Data](#332-audio-data)
  - [3.4 Domain Specific Applications](#34-domain-specific-applications)
    - [3.4.1 Spatial-Temporal Data](#341-spatial-temporal-data)
    - [3.4.2 Medical Time Series](#342-medical-time-series)
    - [3.4.3 Financial Time Series](#343-financial-time-series)
  - [3.5 Gaps and Outlooks](#35-gaps-and-outlooks)
    - [3.5.1 Reuse Which Modality](#351-reuse-which-modality)
- [4. Datasets for Multi-Modal Time Series Analysis](#4-datasets-for-multi-modal-time-series-analysis)
  - [4.1 General Datasets](#41-general-datasets)
  - [4.2 Financial Datasets](#42-financial-datasets)
  - [4.3 Medical Datasets](#43-medical-datasets)
  - [4.4 Spatial-Temporal Datasets](#44-spatial-temporal-datasets)

## 1. Time2X and X2Time

### 1.1 Text to Time Series

#### 1.1.1 Generation

<a id="111-generation"></a>

| Title | Venue |
|-------|-------|
| Language Models Still Struggle to Zero-shot Reason about Time Series | EMNLP 2024 Findings |
| DiffuSETS: 12-lead ECG Generation Conditioned on Clinical Text Reports and Patient-Specific Information | arXiv 25.01 |
| ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning | arXiv 24.12 |

#### 1.1.2 Retrieval

<a id="112-retrieval"></a>

| Title | Venue |
|-------|-------|
| Evaluating Large Language Models on Time Series Feature Understanding: A Comprehensive Taxonomy and Benchmark | EMNLP 2024 |
| TimeSeriesExam: A Time Series Understanding Exam | NeurIPS 2024 Workshop on Time Series in the Age of Large Models |
| CLaSP: Learning Concepts for Time-Series Signals from Natural Language Supervision | arXiv 24.11 |

### 1.2 Time Series to Text

#### 1.2.1 Explanation

<a id="121-explanation"></a>

| Title | Venue |
|-------|-------|
| Inferring Event Descriptions from Time Series with Language Models | arXiv 25.03 |
| Explainable Multi-modal Time Series Prediction with LLM-in-the-Loop | arXiv 25.03 |
| Xforecast: Evaluating natural language explanations for time series forecasting | arXiv 24.10 |
| Large language models can deliver accurate and interpretable time series anomaly detection | arXiv 24.05 |

#### 1.2.2 Captioning

<a id="122-captioning"></a>

| Title | Venue |
|-------|-------|
| Repr2Seq: A Data-to-Text Generation Model for Time Series | IJCNN 2023 |
| Insight miner: A time series analysis dataset for cross-domain alignment with natural language | NeurIPS 2023 AI for Science Workshop |
| T 3: Domain-agnostic neural time-series narration | ICDM 2021 |
| Neural data-driven captioning of time-series line charts | Proceedings of the 2020 International Conference on Advanced Visual Interfaces |
| Time Series Language Model for Descriptive Caption Generation | arXiv 25.01 |
| Decoding Time Series with LLMs: A Multi-Agent Framework for Cross-Domain Annotation | arXiv 24.10 |
| Domain-Independent Automatic Generation of Descriptive Texts for Time-Series Data | arXiv 24.09 |

### 1.3 Text to Time + Time to Text

<a id="13-text-to-time-+-time-to-text"></a>

| Title | Venue |
|-------|-------|
| ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data | AAAI 2025 |
| Time-MQA: Time Series Multi-Task Question Answering with Context Enhancement | arXiv 25.03 |
| ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning | arXiv 24.12 |
| Multi-Modal Forecaster: Jointly Predicting Time Series and Textual Data | arXiv 24.11 |

### 1.4 Other Cross-Modality Works

<a id="14-other-cross-modality-works"></a>

| Title | Venue |
|-------|-------|
| DataNarrative: Automated Data-Driven Storytelling with Visualizations and Texts | EMNLP 2024 |

### 1.5 Domain Specific Applications

#### 1.5.1 Spatial-Temporal Data

<a id="151-spatial-temporal-data"></a>

| Title | Venue |
|-------|-------|
| Urbanclip: Learning text-enhanced urban region profiling with contrastive language-image pretraining from the web | the Web Conference 2024 |
| Urbangpt: Spatio-temporal large language models | KDD 2024 |
| Research on the visualization of spatio-temporal data | IOP Conference Series: Earth and Environmental Science |
| Spatial temporal data visualization in emergency management: a view from data-driven decision | Proceedings of the 3rd ACM SIGSPATIAL International Workshop on the Use of GIS in Emergency Management |
| Teochat: A large vision-language assistant for temporal earth observation data | arXiv 24.10 |

#### 1.5.2 Medical Time Series

<a id="152-medical-time-series"></a>

| Title | Venue |
|-------|-------|
| Electrocardiogram Report Generation and Question Answering via Retrieval-Augmented Self-Supervised Modeling | NeurIPS 2024 Workshop on Time Series in the Age of Large Models |
| Frozen language model helps ecg zero-shot learning | Medical Imaging with Deep Learning |
| Multimodal Models for Comprehensive Cardiac Diagnostics via ECG Interpretation | IEEE International Conference on Bioinformatics and Biomedicine (BIBM) 2024 |
| ECG Semantic Integrator (ESI): A Foundation ECG Model Pretrained with LLM-Enhanced Cardiological Text | Transactions on Machine Learning Research |
| Towards a Personal Health Large Language Model | Advancements In Medical Foundation Models: Explainability, Robustness, Security, and Beyond |
| Diffusion-based conditional ECG generation with structured state space models | Computers in biology and medicine |
| Text-to-ecg: 12-lead electrocardiogram synthesis conditioned on clinical text reports | ICASSP 2023 |
| From data to text in the neonatal intensive care unit: Using NLG technology for decision support and information management | Ai Communications |
| Using natural language generation technology to improve information flows in intensive care units | ECAI 2008 |
| Summarising complex ICU data in natural language | AMIA annual symposium proceedings |
| DiffuSETS: 12-lead ECG Generation Conditioned on Clinical Text Reports and Patient-Specific Information | arXiv 25.01 |
| Automated medical report generation for ecg data: Bridging medical text and signal processing with deep learning | arXiv 24.12 |
| ECG-Chat: A Large ECG-Language Model for Cardiac Disease Diagnosis | arXiv 24.08 |
| Medtsllm: Leveraging llms for multimodal medical time series analysis | arXiv 24.08 |
| MEIT: Multi-modal electrocardiogram instruction tuning on large language models for report generation | arXiv 24.03 |
| Electrocardiogram instruction tuning for report generation | arXiv 2024.03 |
| BioSignal Copilot: Leveraging the power of LLMs in drafting reports for biomedical signals | medRxiv 23.06 |

#### 1.5.3 Financial Time Series

<a id="153-financial-time-series"></a>

| Title | Venue |
|-------|-------|
| Knowledge-augmented Financial Market Analysis and Report Generation | EMNLP 2024 Industry Track |
| FinTral: A Family of GPT-4 Level Multimodal Financial Large Language Models | ACL 2024 Findings|
| Long Text and Multi-Table Summarization: Dataset and Method | EMNLP 2022 Findings |
| Neural abstractive summarization for long text and multiple tables | IEEE Transactions on Knowledge and Data Engineering |
| Open-finllms: Open multimodal large language models for financial applications | arXiv 24.08 |
| Multimodal gen-ai for fundamental investment research | arXiv 24.01 |

### 1.6 Gaps and Outlooks

<a id="16-gaps-and-outlooks"></a>

#### 1.6.1 Unseen Tasks: Introducing Reasoning

<a id="unseen-tasks-papers-1"></a>

| Title | Venue |
|-------|-------|
| A picture is worth a thousand numbers: Enabling llms reason about time series via visualization | NAACL 2025 |
| Evaluating System 1 vs. 2 Reasoning Approaches for Zero-Shot Time-Series Forecasting: A Benchmark and Insights | arXiv 25.03 |
| Position: Empowering Time Series Reasoning with Multimodal LLMs | arXiv 25.02 |
| ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning | arXiv 24.12 |
| Beyond Forecasting: Compositional Time Series Reasoning for End-to-End Task Execution | arXiv 24.10 |
| Towards Time Series Reasoning with LLMs | arXiv 24.09 |

## 2. Time+X

### 2.1 Time Series + Text

*we list papers using dynamic text especially with exogenous information
<a id="21-time-series-+-text"></a>

#### Forecasting (& Imputation)
| Title | Venue |
|-------|-------|
| ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data | AAAI 2025 |
| Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis | NeurIPS 2024 |
| From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection | NeurIPS 2024 |
| GPT4MTS: Prompt-Based Large Language Model for Multimodal Time-Series Forecasting | AAAI 2024 |
| Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative | arXiv 25.02 |
| Context is Key: A Benchmark for Forecasting with Essential Textual Information | arXiv 24.10 |
| Beyond trend and periodicity: Guiding time series forecasting with textual cues | arXiv 24.05 |
| Dual-Forecaster: A Multimodal Time Series Model Integrating Descriptive and Predictive Texts | Openreview |

#### Classification
| Title | Venue |
|-------|-------|
| Advancing time series classification with multimodal language modeling | arXiv 24.03 |
| Hierarchical Multimodal LLMs with Semantic Space Alignment for Enhanced Time Series Classification | arXiv 24.10 |
| Dualtime: A dual-adapter multimodal language model for time series representation | arXiv 24.06 |

### 2.2 Time Series + Other Modalities

<a id="22-time-series-+-other-modalities"></a>

| Title | Venue |
|-------|-------|
| Imagebind: One embedding space to bind them all | CVPR 2023 |
| LANISTR: Multimodal learning from structured and unstructured data | arXiv 23.05 |

### 2.3 Domain Specific Applications

#### 2.3.1 Spatial-Temporal Data

<a id="231-spatial-temporal-data"></a>

| Title | Venue |
|-------|-------|
| Terra: A Multimodal Spatio-Temporal Dataset Spanning the Earth | NeurIPS 2024 |
| BjTT: A large-scale multimodal dataset for traffic prediction | IEEE Transactions on Intelligent Transportation Systems |
| Event Traffic Forecasting with Sparse Multimodal Data | Proceedings of the 32nd ACM International Conference on Multimedia 2024 |
| Mmst-vit: Climate change-aware crop yield prediction via multi-modal spatial-temporal vision transformer | CVPR 2023 |
| Mobile traffic prediction in consumer applications: A multimodal deep learning approach | IEEE Transactions on Consumer Electronics |
| Urban informal settlements classification via a transformer-based spatial-temporal fusion network using multimodal remote sensing and time-series human activity data | International Journal of Applied Earth Observation and Geoinformation |
| Spatial-temporal attention-based convolutional network with text and numerical information for stock price prediction | Neural Computing and Applications |
| Traffic congestion prediction using toll and route search log data | IEEE International Conference on Big Data (Big Data) 2022 |
| Understanding city traffic dynamics utilizing sensor and textual observations | AAAI 2016 |
| Citygpt: Empowering urban spatial cognition of large language models | arXiv 24.06 |
| Where Would I Go Next? Large Language Models as Human Mobility Predictors | arXiv 23.08 |
| Leveraging Language Foundation Models for Human Mobility Forecasting | arXiv 22.09 |

#### 2.3.2 Medical Time Series

<a id="232-medical-time-series"></a>

| Title | Venue |
|-------|-------|
| Addressing asynchronicity in clinical multimodal fusion via individualized chest x-ray generation | NeurIPS 2024 |
| EMERGE: Enhancing Multimodal Electronic Health Records Predictive Modeling with Retrieval-Augmented Generation | CIKM 2024 |
| Improving medical predictions by irregular multimodal electronic health records modeling | ICML 2023 |
| Multimodal pretraining of medical time series and notes | Machine Learning for Health (ML4H) 2023 |
| Learning missing modal electronic health records with unified multi-modal data embedding and modality-aware attention | Machine Learning for Health (ML4H) 2023 |
| MedFuse: Multi-modal fusion with clinical time-series data and chest X-ray images | Machine Learning for Health (ML4H) 2022 |
| Miracle: Causally-aware imputation via learning missing data mechanisms | NeurIPS 2021 |
| How to leverage the multimodal EHR data for better medical prediction? | Conference on Empirical Methods in Natural Language Processing 2021 |
| Deep multi-modal intermediate fusion of clinical record and time series data in mortality prediction | Frontiers in Molecular Biosciences |
| Integrated multimodal artificial intelligence framework for healthcare applications | NPJ digital medicine |
| PTB-XL, a large publicly available electrocardiography dataset | Scientific data |
| Fusion of medical imaging and electronic health records using deep learning: a systematic review and implementation guidelines | NPJ digital medicine |
| Arbitrary Data as Images: Fusion of Patient Data Across Modalities and Irregular Intervals with Vision Transformers | arXiv 25.01 |
| Towards Predicting Temporal Changes in a Patient's Chest X-ray Images based on Electronic Health Records | arXiv 24.09 |
| Multimodal risk prediction with physiological signals, medical images and clinical notes | medrxiv 23.05 |

#### 2.3.3 Financial Time Series

<a id="233-financial-time-series"></a>

| Title | Venue |
|-------|-------|
| Fnspid: A comprehensive financial news dataset in time series | KDD 2024 |
| Multi-modal deep learning for credit rating prediction using text and numerical data streams | Applied Soft Computing |
| Multimodal multiscale dynamic graph convolution networks for stock price prediction | Pattern Recognition |
| Multi-Modal Financial Time-Series Retrieval Through Latent Space Projections | Proceedings of the Fourth ACM International Conference on AI in Finance |
| Natural language based financial forecasting: a survey | Artificial Intelligence Review |
| Financial analysis, planning & forecasting: Theory and application | Unknown |
| Text2timeseries: Enhancing financial forecasting through time series prediction updates with event-driven insights from large language models | arXiv 24.07 |
| Natural language processing and multimodal stock price prediction | arXiv 24.01 |
| Modality-aware Transformer for Financial Time series Forecasting | arXiv 23.10 |
| Predicting financial market trends using time series analysis and natural language processing | arXiv 23.09 |
| Stock price prediction using sentiment analysis and deep learning for Indian markets | arXiv 22.04 |
| Volatility prediction using financial disclosures sentiments with word embedding-based IR models | arXiv 17.02 |

### 2.4 Gaps and Outlooks

<a id="24-gaps-and-outlooks"></a>

#### 2.4.1 Heterogeneous Modality Combinations

<a id="heterogeneous-modality-combinations-papers-2"></a>

| Title | Venue |
|-------|-------|
| Imagebind: One embedding space to bind them all | CVPR 2023 |
| LANISTR: Multimodal learning from structured and unstructured data | arXiv 23.05 |

## 3. TimeAsX

### 3.1 Time Series as Text

<a id="31-time-series-as-text"></a>

| Title | Venue |
|-------|-------|
| Context-Alignment: Activating and Enhancing LLM Capabilities in Time Series | ICLR 2025 |
| ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data | AAAI 2025 |
| Exploiting Language Power for Time Series Forecasting with Exogenous Variables | THE WEB CONFERENCE 2025 |
| Lstprompt: Large language models as zero-shot time series forecasters by long-short-term prompting | ACL 2024 Findings |
| TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting | ICLR 2024 |
| TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series | ICLR 2024 |
| Time-LLM: Time Series Forecasting by Reprogramming Large Language Models | ICLR 2024 |
| Are language models actually useful for time series forecasting? | NeurIPS 2024 |
| Autotimes: Autoregressive time series forecasters via large language models | NeurIPS 2024 |
| S2 IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting | ICML 2024 |
| Large language models are zero-shot time series forecasters | NeurIPS 2023 |
| One fits all: Power general time series analysis by pretrained lm | NeurIPS 2023 |
| PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting | IEEE Transactions on Knowledge and Data Engineering |
| Chronos: Learning the language of time series | TMLR |
| LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters | ACM Transactions on Intelligent Systems and Technology |
| Large Language Models are Few-shot Multivariate Time Series Classifiers | arXiv 25.02 |
| TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with Large Language Model Agents | arXiv 25.02 |
| ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning | arXiv 24.12 |
| Large language models can deliver accurate and interpretable time series anomaly detection | arXiv 24.05 |
| Multi-Patch Prediction: Adapting LLMs for Time Series Representation Learning | arXiv 24.02 |
| Lag-llama: Towards foundation models for time series forecasting | arXiv 23.10 |

### 3.2 Time Series as Image

<a id="32-time-series-as-image"></a>

| Title | Venue |
|-------|-------|
| CAFO: Feature-Centric Explanation on Time Series Classification | KDD 2024 |
| TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis | ICLR 2024 |
| Towards total recall in industrial anomaly detection | CVPR 2022 |
| Deep video prediction for time series forecasting | Proceedings of the Second ACM International Conference on AI in Finance 2021 |
| Forecasting with time series imaging | Expert Systems with Applications |
| Can Multimodal LLMs Perform Time Series Anomaly Detection? | arXiv 25.02 |
| Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting | arXiv 25.02 |
| See it, Think it, Sorted: Large Multimodal Models are Few-shot Time Series Anomaly Analyzers | arXiv 24.11 |
| Plots Unlock Time-Series Understanding in Multimodal Models | arXiv 24.10 |
| VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters | arXiv 24.08 |
| Training-Free Time-Series Anomaly Detection: Leveraging Image Foundation Models | arXiv 24.08 |
| ViTime: A Visual Intelligence-Based Foundation Model for Time Series Forecasting | arXiv 24.07 |
| Time Series as Images: Vision Transformer for Irregularly Sampled Time Series | arXiv 23.03 |
| An image is worth 16x16 words: Transformers for image recognition at scale | arXiv 20.10 |
| Imaging Time-Series to Improve Classification and Imputation | arXiv 15.06 |

### 3.3 Time Series as Other Modalities

#### 3.3.1 Tabular Data

<a id="331-tabular-data"></a>

| Title | Venue |
|-------|-------|
| Forecastpfn: Synthetically-trained zero-shot forecasting | NeurIPS 2024 |
| The Tabular Foundation Model TabPFN Outperforms Specialized Time Series Forecasting Models Based on Simple Features | NeurIPS 2024 Third Table Representation Learning Workshop |
| TableTime: Reformulating Time Series Classification as Zero-Shot Table Understanding via Large Language Models | arXiv 24.11 |
| Tabular Transformers for Modeling Multivariate Time Series | arXiv 20.11 |

#### 3.3.2 Audio Data

<a id="332-audio-data"></a>

| Title | Venue |
|-------|-------|
| Ssast: Self-supervised audio spectrogram transformer | AAAI 2022 |
| T-wavenet: a tree-structured wavelet neural network for time series signal analysis | ICLR 2022 |
| Voice2Series: Reprogramming Acoustic Models for Time Series Classification | arXiv 21.06 |

### 3.4 Domain Specific Applications

#### 3.4.1 Spatial-Temporal Data

<a id="341-spatial-temporal-data"></a>

| Title | Venue |
|-------|-------|
| Spatial-temporal large language model for traffic prediction | 25th IEEE International Conference on Mobile Data Management (MDM) 2024 |
| Unist: A prompt-empowered universal model for urban spatio-temporal prediction | KDD 2024 |
| Urbangpt: Spatio-temporal large language models | KDD 2024 |
| Vmrnn: Integrating vision mamba and lstm for efficient and accurate spatiotemporal forecasting | CVPR 2024 |
| Learning social meta-knowledge for nowcasting human mobility in disaster | the Web Conference 2023 |
| Storm-gan: spatio-temporal meta-gan for cross-city estimation of human mobility responses to covid-19 | ICDM 2022 |
| Deep multi-view spatial-temporal network for taxi demand prediction | AAAI 2018 |
| Trafficgpt: Viewing, processing and interacting with traffic foundation models | Transport Policy |
| Deep spatio-temporal adaptive 3d convolutional neural networks for traffic flow prediction | ACM Transactions on Intelligent Systems and Technology (TIST) |
| ClimateLLM: Efficient Weather Forecasting via Frequency-Aware Large Language Models | arXiv 25.02 |
| TPLLM: A traffic prediction framework based on pretrained large language models | arXiv 24.03 |
| How can large language models understand spatial-temporal data? | arXiv 24.01 |

#### 3.4.2 Medical Time Series

<a id="342-medical-time-series"></a>

| Title | Venue |
|-------|-------|
| ECG-LLM: Leveraging Large Language Models for Low-Quality ECG Signal Restoration | IEEE International Conference on Bioinformatics and Biomedicine (BIBM) 2024 |
| Multimodal llms for health grounded in individual-specific data | Workshop on Machine Learning for Multimodal Healthcare Data 2023 |
| ECG-Chat: A Large ECG-Language Model for Cardiac Disease Diagnosis | arXiv 24.08 |
| Medtsllm: Leveraging llms for multimodal medical time series analysis | arXiv 24.08 |
| Dualtime: A dual-adapter multimodal language model for time series representation | arXiv 24.06 |
| Large language models are few-shot health learners | arXiv 23.05 |

#### 3.4.3 Financial Time Series

<a id="343-financial-time-series"></a>

| Title | Venue |
|-------|-------|
| MTRGL: Effective Temporal Correlation Discerning through Multi-modal Temporal Relational Graph Learning | ICASSP 2024 |
| From pixels to predictions: Spectrogram and vision transformer for better time series forecasting | Proceedings of the Fourth ACM International Conference on AI in Finance 2023 |
| Quantum-enhanced forecasting: Leveraging quantum gramian angular field and CNNs for stock return predictions | Finance Research Letters |
| Deep learning-based spatial-temporal graph neural networks for price movement classification in crude oil and precious metal markets | Machine Learning with Applications |
| Financial time series forecasting with multi-modality graph neural network | Pattern Recognition |
| Encoding candlesticks as images for pattern classification using convolutional neural networks | Financial Innovation |
| Forecasting with time series imaging | Expert Systems with Applications |
| Research on financial multi-asset portfolio risk prediction model based on convolutional neural networks and image processing | arXiv 24.12 |
| A Stock Price Prediction Approach Based on Time Series Decomposition and Multi-Scale CNN using OHLCT Images | arXiv 24.10 |
| An image is worth 16x16 words: Transformers for image recognition at scale | arXiv 20.10 |
| Image processing tools for financial time series classification | arXiv 20.08 |
| Financial trading model with stock bar chart image time series with deep convolutional neural networks | arXiv 19.03 |
| Imaging time-series to improve classification and imputation | arXiv 15.06 |

### 3.5 Gaps and Outlooks

<a id="35-gaps-and-outlooks"></a>

#### 3.5.1 Reuse Which Modality

<a id="351-reuse-which-modality"></a>

| Title | Venue |
|-------|-------|
| A picture is worth a thousand numbers: Enabling llms reason about time series via visualization | NAACL25 |
| Can LLMs Understand Time Series Anomalies? | ICLR 2025 |
| Vision-Enhanced Time Series Forecasting via Latent Diffusion Models | arXiv 25.02 |

## 4. Datasets for Multi-Modal Time Series Analysis

### 4.1 General Datasets

<a id="41-general-datasets"></a>

| Dataset | Modalities | Highlights |
|---------|------------|------------|
| [Time-MMD](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8e7768122f3eeec6d77cd2b424b72413-Abstract-Datasets_and_Benchmarks_Track.html) | Time+Text | 9 Domains; Real Datasets (general context); Across More than 24 Years |
| [ChatTime](https://arxiv.org/abs/2412.11376) | Time+Text | 3 Real Datasets (weather&date) |
| [CiK](https://arxiv.org/abs/2410.18959) | Time+Text | 7 Domains; 71 Human-Designed Tasks |
| [ChatTS](https://arxiv.org/abs/2412.03104) | Time+Text | Synthetic Method; 500+ Human-Labeled Samples |
| [TSQA](https://www.arxiv.org/abs/2503.01875) | Time+Text | Multi-Task QA Format; 1.4k Human-Selected Samples |

### 4.2 Financial Datasets

<a id="42-financial-datasets"></a>

| Dataset | Modalities | Highlights |
|---------|------------|------------|
| [FNSPID](https://dl.acm.org/doi/abs/10.1145/3637528.3671629) | Time+Text | Large-Scale Finance News; Across 24 Years |
| [FinBen](https://proceedings.neurips.cc/paper_files/paper/2024/hash/adb1d9fa8be4576d28703b396b82ba1b-Abstract-Datasets_and_Benchmarks_Track.html) | Time+Text | Bilingual; 42 Sub-Datasets; 8 Tasks |

### 4.3 Medical Datasets

<a id="43-medical-datasets"></a>

| Dataset | Modalities | Highlights |
|---------|------------|------------|
| [MIMIC](https://paperswithcode.com/dataset/mimic-iii) | Time+Text+Image+Table | Multiple Medical Tasks; Expert-Labeled Data |
| [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) | Time+Text | Large-Scale Expert-Labeled ECG Data |

### 4.4 Spatial-Temporal Datasets

<a id="44-spatial-temporal-datasets"></a>

| Dataset | Modalities | Highlights |
|---------|------------|------------|
| [CityEval](https://arxiv.org/abs/2406.13948) | Time+Text+Image | Multiple Urban Tasks, capable of Involving LLMs |
| [Terra](https://papers.nips.cc/paper_files/paper/2024/hash/7a6a7fbd1ee0c9684b3f919f79d129ef-Abstract-Datasets_and_Benchmarks_Track.html) | Time+Text+Image | Worldwide Grid Data across 45 Years |

## Contact
If you have any questions or suggestions, feel free to contact:
hliu763@gatech.edu
## Acknowledgement

We refer to the following repos:

https://github.com/ForestsKing/Awesome-Multimodal-Time-Series

https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM



