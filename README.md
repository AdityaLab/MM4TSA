# Awesome Multi-Modal Time Series Analysis Papers (MM4TSA)

A curated list of papers in the emerging field of multi-modal time series analysis, where multiple data modalities are combined to enhance time series analysis tasks.

**Sorting Method**: Papers in each table are sorted by publication venue first (published papers take precedence over preprints), then by recency (newer arXiv preprints appear before older ones).

## Table of Contents

- [1. Time2X and X2Time](#1-time2x-and-x2time)
  - [1.1 Text to Time Series](#11-text-to-time-series)
    - [1.1.1 Generation](#111-generation) - [Jump to table](#generation-papers)
    - [1.1.2 Retrieval](#112-retrieval) - [Jump to table](#retrieval-papers)
  - [1.2 Time Series to Text](#12-time-series-to-text)
    - [1.2.1 Explanation](#121-explanation) - [Jump to table](#explanation-papers)
    - [1.2.2 Captioning](#122-captioning) - [Jump to table](#captioning-papers)
  - [1.3 Text to Time + Time to Text](#13-text-to-time-+-time-to-text) - [Jump to table](#text-to-time--time-to-text-papers)
  - [1.4 Domain Specific Applications](#14-domain-specific-applications)
    - [1.4.1 Spatial-Temporal Data](#141-spatial-temporal-data) - [Jump to table](#spatial-temporal-data-papers-1)
    - [1.4.2 Medical Time Series](#142-medical-time-series) - [Jump to table](#medical-time-series-papers-1)
    - [1.4.3 Financial Time Series](#143-financial-time-series) - [Jump to table](#financial-time-series-papers-1)
  - [1.5 Gaps and Outlooks](#15-gaps-and-outlooks) - [Jump to table](#gaps-and-outlooks-papers)
- [2. Time+X](#2-time+x)
  - [2.1 Time Series + Text](#21-time-series-+-text) - [Jump to table](#time-series--text-papers)
  - [2.2 Time Series + Other Modalities](#22-time-series-+-other-modalities) - [Jump to table](#time-series--other-modalities-papers)
  - [2.3 Domain Specific Applications](#23-domain-specific-applications)
    - [2.3.1 Spatial-Temporal Data](#231-spatial-temporal-data) - [Jump to table](#spatial-temporal-data-papers-2)
    - [2.3.2 Medical Time Series](#232-medical-time-series) - [Jump to table](#medical-time-series-papers-2)
    - [2.3.3 Financial Time Series](#233-financial-time-series) - [Jump to table](#financial-time-series-papers-2)
  - [2.4 Gaps and Outlooks](#24-gaps-and-outlooks) - [Jump to table](#gaps-and-outlooks-papers)
- [3. TimeAsX](#3-timeasx)
  - [3.1 Time Series as Text](#31-time-series-as-text) - [Jump to table](#time-series-as-text-papers)
  - [3.2 Time Series as Image](#32-time-series-as-image) - [Jump to table](#time-series-as-image-papers)
  - [3.3 Time Series as Other Modalities](#33-time-series-as-other-modalities)
    - [3.3.1 Tabular Data](#331-tabular-data) - [Jump to table](#tabular-data-papers)
    - [3.3.2 Audio Data](#332-audio-data) - [Jump to table](#audio-data-papers)
  - [3.4 Domain Specific Applications](#34-domain-specific-applications)
    - [3.4.1 Spatial-Temporal Data](#341-spatial-temporal-data) - [Jump to table](#spatial-temporal-data-papers-3)
    - [3.4.2 Medical Time Series](#342-medical-time-series) - [Jump to table](#medical-time-series-papers-3)
    - [3.4.3 Financial Time Series](#343-financial-time-series) - [Jump to table](#financial-time-series-papers-3)
  - [3.5 Gaps and Outlooks](#35-gaps-and-outlooks) - [Jump to table](#gaps-and-outlooks-papers)
- [4. Representative Multi-Modal Time-Series Datasets](#4-datasets-for-multi-modal-time-series-analysis)
  - [4.1 General Datasets](#41-general-datasets) - [Jump to table](#general-datasets-table)
  - [4.2 Financial Datasets](#42-financial-datasets) - [Jump to table](#financial-datasets-table)
  - [4.3 Medical Datasets](#43-medical-datasets) - [Jump to table](#medical-datasets-table)
  - [4.4 Spatial-Temporal Datasets](#44-spatial-temporal-datasets) - [Jump to table](#spatial-temporal-datasets-table)

## 1. Time2X and X2Time

### 1.1 Text to Time Series

#### 1.1.1 Generation

<a id='generation-papers-1'></a>

| Title | Venue |
|-------|-------|
| Language Models Still Struggle to Zero-shot Reason about Time Series | EMNLP (Findings) |
| DiffuSETS: 12-lead ECG Generation Conditioned on Clinical Text Reports and Patient-Specific Information | arXiv 25.01 |
| ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning | arXiv 24.12 |

#### 1.1.2 Retrieval

<a id='retrieval-papers-1'></a>

| Title | Venue |
|-------|-------|
| Evaluating Large Language Models on Time Series Feature Understanding: A Comprehensive Taxonomy and Benchmark | Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing |
| TimeSeriesExam: A Time Series Understanding Exam | NeurIPS Workshop on Time Series in the Age of Large Models |
| CLaSP: Learning Concepts for Time-Series Signals from Natural Language Supervision | arXiv 24.11 |

### 1.2 Time Series to Text

#### 1.2.1 Explanation

<a id='explanation-papers-1'></a>

| Title | Venue |
|-------|-------|
| Explainable Multi-modal Time Series Prediction with LLM-in-the-Loop | arXiv 25.03 |
| Xforecast: Evaluating natural language explanations for time series forecasting | arXiv 24.10 |
| Large language models can deliver accurate and interpretable time series anomaly detection | arXiv 24.05 |

#### 1.2.2 Captioning

<a id='captioning-papers-1'></a>

| Title | Venue |
|-------|-------|
| Repr2Seq: A Data-to-Text Generation Model for Time Series | 2023 International Joint Conference on Neural Networks (IJCNN) |
| Insight miner: A time series analysis dataset for cross-domain alignment with natural language | NeurIPS 2023 AI for Science Workshop |
| T 3: Domain-agnostic neural time-series narration | 2021 IEEE International Conference on Data Mining (ICDM) |
| Neural data-driven captioning of time-series line charts | Proceedings of the 2020 International Conference on Advanced Visual Interfaces |
| Time Series Language Model for Descriptive Caption Generation | arXiv 25.01 |
| Domain-Independent Automatic Generation of Descriptive Texts for Time-Series Data | arXiv 24.09 |
| Decoding Time Series with LLMs: A Multi-Agent Framework for Cross-Domain Annotation | arXiv 24.10 |

### 1.3 Text to Time + Time to Text

<a id='text-to-time--time-to-text-papers'></a>

| Title | Venue |
|-------|-------|
| ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data | AAAI |
| DataNarrative: Automated Data-Driven Storytelling with Visualizations and Texts | Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing |
| Time-MQA: Time Series Multi-Task Question Answering with Context Enhancement | arXiv 25.03 |
| ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning | arXiv 24.12 |
| Multi-Modal Forecaster: Jointly Predicting Time Series and Textual Data | arXiv 24.11 |

### 1.4 Domain Specific Applications

#### 1.4.1 Spatial-Temporal Data

<a id='spatial-temporal-data-papers-1'></a>

| Title | Venue |
|-------|-------|
| Urbanclip: Learning text-enhanced urban region profiling with contrastive language-image pretraining from the web | Proceedings of the ACM Web Conference 2024 |
| Urbangpt: Spatio-temporal large language models | Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining |
| Research on the visualization of spatio-temporal data | IOP Conference Series: Earth and Environmental Science |
| Spatial temporal data visualization in emergency management: a view from data-driven decision | Proceedings of the 3rd ACM SIGSPATIAL International Workshop on the Use of GIS in Emergency Management |
| Teochat: A large vision-language assistant for temporal earth observation data | arXiv 24.10 |

#### 1.4.2 Medical Time Series

<a id='medical-time-series-papers-1'></a>

| Title | Venue |
|-------|-------|
| Electrocardiogram Report Generation and Question Answering via Retrieval-Augmented Self-Supervised Modeling | NeurIPS Workshop on Time Series in the Age of Large Models |
| Frozen language model helps ecg zero-shot learning | Medical Imaging with Deep Learning |
| Multimodal Models for Comprehensive Cardiac Diagnostics via ECG Interpretation | 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM) |
| ECG Semantic Integrator (ESI): A Foundation ECG Model Pretrained with LLM-Enhanced Cardiological Text | Transactions on Machine Learning Research |
| Towards a Personal Health Large Language Model | Advancements In Medical Foundation Models: Explainability, Robustness, Security, and Beyond |
| Diffusion-based conditional ECG generation with structured state space models | Computers in biology and medicine |
| Text-to-ecg: 12-lead electrocardiogram synthesis conditioned on clinical text reports | ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) |
| BioSignal Copilot: Leveraging the power of LLMs in drafting reports for biomedical signals | medRxiv |
| From data to text in the neonatal intensive care unit: Using NLG technology for decision support and information management | Ai Communications |
| Using natural language generation technology to improve information flows in intensive care units | ECAI 2008 |
| Summarising complex ICU data in natural language | AMIA annual symposium proceedings |
| DiffuSETS: 12-lead ECG Generation Conditioned on Clinical Text Reports and Patient-Specific Information | arXiv 25.01 |
| Automated medical report generation for ecg data: Bridging medical text and signal processing with deep learning | arXiv 24.12 |
| MEIT: Multi-modal electrocardiogram instruction tuning on large language models for report generation | arXiv 24.03 |
| Electrocardiogram instruction tuning for report generation | arXiv 2024 |
| ECG-Chat: A Large ECG-Language Model for Cardiac Disease Diagnosis | arXiv 24.08 |
| Medtsllm: Leveraging llms for multimodal medical time series analysis | arXiv 24.08 |

#### 1.4.3 Financial Time Series

<a id='financial-time-series-papers-1'></a>

| Title | Venue |
|-------|-------|
| Knowledge-augmented Financial Market Analysis and Report Generation | Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track |
| FinTral: A Family of GPT-4 Level Multimodal Financial Large Language Models | Findings of the Association for Computational Linguistics ACL 2024 |
| Neural abstractive summarization for long text and multiple tables | IEEE Transactions on Knowledge and Data Engineering |
| Diffusion-based conditional ECG generation with structured state space models | Computers in biology and medicine |
| Long Text and Multi-Table Summarization: Dataset and Method | Findings of the Association for Computational Linguistics: EMNLP 2022 |
| Open-finllms: Open multimodal large language models for financial applications | arXiv 24.08 |
| Multimodal gen-ai for fundamental investment research | arXiv 24.01 |

### 1.5 Gaps and Outlooks

#### 1.5.1 Unseen Tasks

<a id='unseen-tasks-papers-1'></a>

| Title | Venue |
|-------|-------|
| A picture is worth a thousand numbers: Enabling llms reason about time series via visualization | NAACL25 |
| Evaluating System 1 vs. 2 Reasoning Approaches for Zero-Shot Time-Series Forecasting: A Benchmark and Insights | arXiv 25.03 |
| Beyond Forecasting: Compositional Time Series Reasoning for End-to-End Task Execution | arXiv 24.10 |
| ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning | arXiv 24.12 |

## 2. Time+X

### 2.1 Time Series + Text

<a id='time-series--text-papers'></a>

| Title | Venue |
|-------|-------|
| ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data | AAAI |
| Autotimes: Autoregressive time series forecasters via large language models | Advances in Neural Information Processing Systems |
| Fnspid: A comprehensive financial news dataset in time series | Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining |
| Time-series forecasting for out-of-distribution generalization using invariant learning | Proceedings of the 41st International Conference on Machine Learning |
| Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis | Advances in Neural Information Processing Systems |
| Chain-of-thought prompting elicits reasoning in large language models | Advances in neural information processing systems |
| Dual-Forecaster: A Multimodal Time Series Model Integrating Descriptive and Predictive Texts | Unknown |
| Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative | arXiv 25.02 |
| Lstprompt: Large language models as zero-shot time series forecasters by long-short-term prompting | arXiv 24.02 |
| Context is Key: A Benchmark for Forecasting with Essential Textual Information | arXiv 24.10 |
| Beyond trend and periodicity: Guiding time series forecasting with textual cues | arXiv 24.05 |
| From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection | arXiv 24.09 |
| Advancing time series classification with multimodal language modeling | arXiv 24.03 |
| Hierarchical Multimodal LLMs with Semantic Space Alignment for Enhanced Time Series Classification | arXiv 24.10 |
| Dualtime: A dual-adapter multimodal language model for time series representation | arXiv 24.06 |
| Time-LLM: Time Series Forecasting by Reprogramming Large Language Models | arXiv 23.10 |
| Time-llm: Time series forecasting by reprogramming large language models | arXiv 23.10 |

### 2.2 Time Series + Other Modalities

<a id='time-series--other-modalities-papers'></a>

| Title | Venue |
|-------|-------|
| Imagebind: One embedding space to bind them all | Proceedings of the IEEE/CVF conference on computer vision and pattern recognition |
| LANISTR: Multimodal learning from structured and unstructured data | arXiv 23.05 |

### 2.3 Domain Specific Applications

#### 2.3.1 Spatial-Temporal Data

<a id='spatial-temporal-data-papers-2'></a>

| Title | Venue |
|-------|-------|
| Terra: A Multimodal Spatio-Temporal Dataset Spanning the Earth | Advances in Neural Information Processing Systems |
| BjTT: A large-scale multimodal dataset for traffic prediction | IEEE Transactions on Intelligent Transportation Systems |
| Event Traffic Forecasting with Sparse Multimodal Data | Proceedings of the 32nd ACM International Conference on Multimedia |
| Mobile traffic prediction in consumer applications: A multimodal deep learning approach | IEEE Transactions on Consumer Electronics |
| Mmst-vit: Climate change-aware crop yield prediction via multi-modal spatial-temporal vision transformer | Proceedings of the IEEE/CVF International Conference on Computer Vision |
| Urban informal settlements classification via a transformer-based spatial-temporal fusion network using multimodal remote sensing and time-series human activity data | International Journal of Applied Earth Observation and Geoinformation |
| Spatial-temporal attention-based convolutional network with text and numerical information for stock price prediction | Neural Computing and Applications |
| Traffic congestion prediction using toll and route search log data | 2022 IEEE International Conference on Big Data (Big Data) |
| Understanding city traffic dynamics utilizing sensor and textual observations | Proceedings of the AAAI Conference on Artificial Intelligence |
| Citygpt: Empowering urban spatial cognition of large language models | arXiv 24.06 |

#### 2.3.2 Medical Time Series

<a id='medical-time-series-papers-2'></a>

| Title | Venue |
|-------|-------|
| Multimodal risk prediction with physiological signals, medical images and clinical notes | Heliyon |
| EMERGE: Enhancing Multimodal Electronic Health Records Predictive Modeling with Retrieval-Augmented Generation | Proceedings of the 33rd ACM International Conference on Information and Knowledge Management |
| Addressing asynchronicity in clinical multimodal fusion via individualized chest x-ray generation | Advances in Neural Information Processing Systems |
| Learning missing modal electronic health records with unified multi-modal data embedding and modality-aware attention | Machine Learning for Healthcare Conference |
| Deep multi-modal intermediate fusion of clinical record and time series data in mortality prediction | Frontiers in Molecular Biosciences |
| Multimodal pretraining of medical time series and notes | Machine Learning for Health (ML4H) |
| Improving medical predictions by irregular multimodal electronic health records modeling | International Conference on Machine Learning |
| MedFuse: Multi-modal fusion with clinical time-series data and chest X-ray images | Proceedings of the 7th Machine Learning for Healthcare Conference |
| Integrated multimodal artificial intelligence framework for healthcare applications | NPJ digital medicine |
| How to leverage the multimodal EHR data for better medical prediction? | Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing |
| Miracle: Causally-aware imputation via learning missing data mechanisms | Advances in Neural Information Processing Systems |
| PTB-XL, a large publicly available electrocardiography dataset | Scientific data |
| Fusion of medical imaging and electronic health records using deep learning: a systematic review and implementation guidelines | NPJ digital medicine |
| Arbitrary Data as Images: Fusion of Patient Data Across Modalities and Irregular Intervals with Vision Transformers | arXiv 25.01 |
| Towards Predicting Temporal Changes in a Patient's Chest X-ray Images based on Electronic Health Records | arXiv 24.09 |

#### 2.3.3 Financial Time Series

<a id='financial-time-series-papers-2'></a>

| Title | Venue |
|-------|-------|
| Multi-modal deep learning for credit rating prediction using text and numerical data streams | Applied Soft Computing |
| Multimodal multiscale dynamic graph convolution networks for stock price prediction | Pattern Recognition |
| Fnspid: A comprehensive financial news dataset in time series | Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining |
| Multi-Modal Financial Time-Series Retrieval Through Latent Space Projections | Proceedings of the Fourth ACM International Conference on AI in Finance |
| Natural language based financial forecasting: a survey | Artificial Intelligence Review |
| Financial analysis, planning \& forecasting: Theory and application | Unknown |
| Text2timeseries: Enhancing financial forecasting through time series prediction updates with event-driven insights from large language models | arXiv 24.07 |
| Natural language processing and multimodal stock price prediction | arXiv 24.01 |
| Modality-aware Transformer for Financial Time series Forecasting | arXiv 23.10 |
| Predicting financial market trends using time series analysis and natural language processing | arXiv 23.09 |
| Stock price prediction using sentiment analysis and deep learning for Indian markets | arXiv 22.04 |
| Volatility prediction using financial disclosures sentiments with word embedding-based IR models | arXiv 17.02 |

### 2.4 Gaps and Outlooks

#### 2.4.1 Heterogeneous Modality Combinations

<a id='heterogeneous-modality-combinations-papers-2'></a>

| Title | Venue |
|-------|-------|
| Imagebind: One embedding space to bind them all | Proceedings of the IEEE/CVF conference on computer vision and pattern recognition |
| LANISTR: Multimodal learning from structured and unstructured data | arXiv 23.05 |

## 3. TimeAsX

### 3.1 Time Series as Text

<a id='time-series-as-text-papers'></a>

| Title | Venue |
|-------|-------|
| ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data | AAAI |
| Exploiting Language Power for Time Series Forecasting with Exogenous Variables | THE WEB CONFERENCE 2025 |
| TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting | The Twelfth International Conference on Learning Representations |
| Autotimes: Autoregressive time series forecasters via large language models | Advances in Neural Information Processing Systems |
| S2 IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting | Forty-first International Conference on Machine Learning |
| Large language models are zero-shot time series forecasters | Advances in Neural Information Processing Systems |
| One fits all: Power general time series analysis by pretrained lm | Advances in neural information processing systems |
| Lag-llama: Towards foundation models for time series forecasting | R0-FoMo: Robustness of Few-shot and Zero-shot Learning in Large Foundation Models |
| Context-Alignment: Activating and Enhancing LLM Capabilities in Time Series | arXiv 25.01 |
| Large Language Models are Few-shot Multivariate Time Series Classifiers | arXiv 25.02 |
| Lstprompt: Large language models as zero-shot time series forecasters by long-short-term prompting | arXiv 24.02 |
| Multi-Patch Prediction: Adapting LLMs for Time Series Representation Learning | arXiv 24.02 |
| Large language models can deliver accurate and interpretable time series anomaly detection | arXiv 24.05 |
| ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning | arXiv 24.12 |
| Chronos: Learning the language of time series | arXiv 24.03 |
| Where Would I Go Next? Large Language Models as Human Mobility Predictors | arXiv 23.08 |
| LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters | arXiv 23.08 |
| TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series | arXiv 23.08 |
| Time-LLM: Time Series Forecasting by Reprogramming Large Language Models | arXiv 23.10 |
| Leveraging Language Foundation Models for Human Mobility Forecasting | arXiv 22.09 |

### 3.2 Time Series as Image

<a id='time-series-as-image-papers'></a>

| Title | Venue |
|-------|-------|
| CAFO: Feature-Centric Explanation on Time Series Classification | Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining |
| Towards total recall in industrial anomaly detection | Proceedings of the IEEE/CVF conference on computer vision and pattern recognition |
| Deep video prediction for time series forecasting | Proceedings of the Second ACM International Conference on AI in Finance |
| Forecasting with time series imaging | Expert Systems with Applications |
| Can Multimodal LLMs Perform Time Series Anomaly Detection? | arXiv 25.02 |
| Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting | arXiv 25.02 |
| ViTime: A Visual Intelligence-Based Foundation Model for Time Series Forecasting | arXiv 24.07 |
| See it, Think it, Sorted: Large Multimodal Models are Few-shot Time Series Anomaly Analyzers | arXiv 24.11 |
| Vitime: A visual intelligence-based foundation model for time series forecasting | arXiv 24.07 |
| Plots Unlock Time-Series Understanding in Multimodal Models | arXiv 24.10 |
| VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters | arXiv 24.08 |
| Training-Free Time-Series Anomaly Detection: Leveraging Image Foundation Models | arXiv 24.08 |
| Time Series as Images: Vision Transformer for Irregularly Sampled Time Series | arXiv 23.03 |
| An image is worth 16x16 words: Transformers for image recognition at scale | arXiv 20.10 |
| Imaging Time-Series to Improve Classification and Imputation | arXiv 15.06 |

### 3.3 Time Series as Other Modalities

#### 3.3.1 Tabular Data

<a id='tabular-data-papers-3'></a>

| Title | Venue |
|-------|-------|
| Forecastpfn: Synthetically-trained zero-shot forecasting | Advances in Neural Information Processing Systems |
| The Tabular Foundation Model TabPFN Outperforms Specialized Time Series Forecasting Models Based on Simple Features | NeurIPS 2024 Third Table Representation Learning Workshop |
| TableTime: Reformulating Time Series Classification as Zero-Shot Table Understanding via Large Language Models | arXiv 24.11 |
| Tabular Transformers for Modeling Multivariate Time Series | arXiv 20.11 |

#### 3.3.2 Audio Data

<a id='audio-data-papers-3'></a>

| Title | Venue |
|-------|-------|
| Ssast: Self-supervised audio spectrogram transformer | Proceedings of the AAAI Conference on Artificial Intelligence |
| T-wavenet: a tree-structured wavelet neural network for time series signal analysis | International conference on learning representations |
| Voice2Series: Reprogramming Acoustic Models for Time Series Classification | arXiv 21.06 |

### 3.4 Domain Specific Applications

#### 3.4.1 Spatial-Temporal Data

<a id='spatial-temporal-data-papers-3'></a>

| Title | Venue |
|-------|-------|
| Trafficgpt: Viewing, processing and interacting with traffic foundation models | Transport Policy |
| Spatial-temporal large language model for traffic prediction | 2024 25th IEEE International Conference on Mobile Data Management (MDM) |
| Unist: A prompt-empowered universal model for urban spatio-temporal prediction | Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining |
| Urbangpt: Spatio-temporal large language models | Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining |
| Vmrnn: Integrating vision mamba and lstm for efficient and accurate spatiotemporal forecasting | Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition |
| Learning social meta-knowledge for nowcasting human mobility in disaster | Proceedings of the ACM Web Conference 2023 |
| Deep spatio-temporal adaptive 3d convolutional neural networks for traffic flow prediction | ACM Transactions on Intelligent Systems and Technology (TIST) |
| Storm-gan: spatio-temporal meta-gan for cross-city estimation of human mobility responses to covid-19 | 2022 IEEE international conference on data mining (ICDM) |
| Deep multi-view spatial-temporal network for taxi demand prediction | Proceedings of the AAAI conference on artificial intelligence |
| ClimateLLM: Efficient Weather Forecasting via Frequency-Aware Large Language Models | arXiv 25.02 |
| How can large language models understand spatial-temporal data? | arXiv 24.01 |
| TPLLM: A traffic prediction framework based on pretrained large language models | arXiv 24.03 |

#### 3.4.2 Medical Time Series

<a id='medical-time-series-papers-3'></a>

| Title | Venue |
|-------|-------|
| ECG-LLM: Leveraging Large Language Models for Low-Quality ECG Signal Restoration | 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM) |
| Multimodal llms for health grounded in individual-specific data | Workshop on Machine Learning for Multimodal Healthcare Data |
| ECG-Chat: A Large ECG-Language Model for Cardiac Disease Diagnosis | arXiv 24.08 |
| Dualtime: A dual-adapter multimodal language model for time series representation | arXiv 24.06 |
| Medtsllm: Leveraging llms for multimodal medical time series analysis | arXiv 24.08 |
| Large language models are few-shot health learners | arXiv 23.05 |

#### 3.4.3 Financial Time Series

<a id='financial-time-series-papers-3'></a>

| Title | Venue |
|-------|-------|
| Quantum-enhanced forecasting: Leveraging quantum gramian angular field and CNNs for stock return predictions | Finance Research Letters |
| MTRGL: Effective Temporal Correlation Discerning through Multi-modal Temporal Relational Graph Learning | ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) |
| Deep learning-based spatial-temporal graph neural networks for price movement classification in crude oil and precious metal markets | Machine Learning with Applications |
| From pixels to predictions: Spectrogram and vision transformer for better time series forecasting | Proceedings of the Fourth ACM International Conference on AI in Finance |
| Financial time series forecasting with multi-modality graph neural network | Pattern Recognition |
| Encoding candlesticks as images for pattern classification using convolutional neural networks | Financial Innovation |
| Forecasting with time series imaging | Expert Systems with Applications |
| A Stock Price Prediction Approach Based on Time Series Decomposition and Multi-Scale CNN using OHLCT Images | arXiv 24.10 |
| Research on financial multi-asset portfolio risk prediction model based on convolutional neural networks and image processing | arXiv 24.12 |
| An image is worth 16x16 words: Transformers for image recognition at scale | arXiv 20.10 |
| Image processing tools for financial time series classification | arXiv 20.08 |
| Financial trading model with stock bar chart image time series with deep convolutional neural networks | arXiv 19.03 |
| Imaging time-series to improve classification and imputation | arXiv 15.06 |

### 3.5 Gaps and Outlooks

#### 3.5.1 Reuse Which Modality

<a id='reuse-which-modality-papers-3'></a>

| Title | Venue |
|-------|-------|
| A picture is worth a thousand numbers: Enabling llms reason about time series via visualization | NAACL25 |
| Vision-Enhanced Time Series Forecasting via Latent Diffusion Models | arXiv 25.02 |

## 4. Datasets for Multi-Modal Time Series Analysis

### 4.1 General Datasets

<a id='general-datasets-table'></a>

| Dataset | Modalities | Highlights |
|---------|------------|------------|
| Time-MMD | Time+Text | 9 Domains; Real Datasets (general context); Across More than 24 Years |
| ChatTime | Time+Text | 3 Real Datasets (weather&date) |
| CiK | Time+Text | 7 Domains; 71 Human-Designed Tasks |
| ChatTS | Time+Text | Synthetic Method; 500+ Human-Labeled Samples |
| TSQA | Time+Text | Multi-Task QA Format; 1.4k Human-Selected Samples |

### 4.2 Financial Datasets

<a id='financial-datasets-table'></a>

| Dataset | Modalities | Highlights |
|---------|------------|------------|
| FNSPID | Time+Text | Large-Scale Finance News; Across 24 Years |
| FinBen | Time+Text | Bilingual; 42 Sub-Datasets; 8 Tasks |

### 4.3 Medical Datasets

<a id='medical-datasets-table'></a>

| Dataset | Modalities | Highlights |
|---------|------------|------------|
| MIMIC | Time+Text+Image+Table | Multiple Medical Tasks; Expert-Labeled Data |
| PTB-XL | Time+Text | Large-Scale Expert-Labeled ECG Data |

### 4.4 Spatial-Temporal Datasets

<a id='spatial-temporal-datasets-table'></a>

| Dataset | Modalities | Highlights |
|---------|------------|------------|
| CityEval | Time+Text+Image | Multiple Urban Tasks, capable of Involving LLMs |
| Terra | Time+Text+Image | Worldwide Grid Data across 45 Years |
