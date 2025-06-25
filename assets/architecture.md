# BMR-Model-Merge Project Architecture

This document describes the detailed architecture of the BMR-model-merge project for Japanese invoice/receipt OCR using evolutionary model merging techniques.

## Overall System Architecture

```mermaid
graph TB
    subgraph "Frontend"
        UI[Demo Web UI]
        CLI[Command Line Interface]
    end
    
    subgraph "Core Optimization"
        GA[Genetic Algorithm]
        BMR[BMR Algorithm]
        BWR[BWR Algorithm]
        MERGE[Model Merging]
    end
    
    subgraph "Models"
        VLM[Vision-Language Models]
        LLM[Language Models]
        OCR[OCR Models]
        FIELD[Field Extraction Models]
    end
    
    subgraph "Data Processing"
        DATASETS[Japanese Datasets]
        PREPROC[Preprocessing Pipeline]
        AUG[Data Augmentation]
    end
    
    subgraph "Evaluation"
        METRICS[Evaluation Metrics]
        VIS[Result Visualization]
        COMPARE[Algorithm Comparison]
    end
    
    UI --> OCR
    CLI --> OCR
    
    GA --> MERGE
    BMR --> MERGE
    BWR --> MERGE
    
    MERGE --> OCR
    
    VLM --> MERGE
    LLM --> MERGE
    
    DATASETS --> PREPROC
    PREPROC --> AUG
    AUG --> GA
    AUG --> BMR
    AUG --> BWR
    
    OCR --> METRICS
    FIELD --> METRICS
    METRICS --> VIS
    METRICS --> COMPARE
```

## Module Relationships

```mermaid
classDiagram
    class OCRModel {
        +load_model()
        +preprocess()
        +postprocess()
        +__call__()
    }
    
    class JapaneseOCRModel {
        +extract_text_from_image()
    }
    
    class InvoiceFieldExtractor {
        +extract_fields()
        +process_ocr_result()
    }
    
    class Dataset {
        +load_dataset()
        +__getitem__()
    }
    
    class JapaneseReceiptDataset {
        +load_dataset()
    }
    
    class JapaneseInvoiceDataset {
        +load_dataset()
    }
    
    class AugmentedOCRDataset {
        +__getitem__()
        +_apply_augmentations()
    }
    
    class Processor {
        +process_text()
        +process_image()
    }
    
    class JapaneseDocumentProcessor {
        +extract_fields()
        +_correct_orientation()
    }
    
    class GeneticOptimizer {
        +evolve_one_generation()
        +evaluate_population()
        +create_offspring()
    }
    
    class BMROptimizer {
        +evaluate_fitness()
        +evolve()
        +get_best_individual()
    }
    
    class BWROptimizer {
        +evaluate_fitness()
        +evolve()
        +get_best_individual()
    }
    
    OCRModel <|-- JapaneseOCRModel
    Dataset <|-- JapaneseReceiptDataset
    Dataset <|-- JapaneseInvoiceDataset
    Dataset <|-- AugmentedOCRDataset
    Processor <|-- JapaneseDocumentProcessor
    
    JapaneseOCRModel --> InvoiceFieldExtractor
    JapaneseOCRModel --> JapaneseDocumentProcessor
    JapaneseReceiptDataset --> JapaneseDocumentProcessor
    JapaneseInvoiceDataset --> JapaneseDocumentProcessor
    
    GeneticOptimizer --> JapaneseOCRModel
    BMROptimizer --> JapaneseOCRModel
    BWROptimizer --> JapaneseOCRModel
```

## Data Workflow

```mermaid
flowchart TD
    A[Japanese Invoice/Receipt Images] --> B[Image Processing]
    B --> C{OCR Processing}
    C --> D[Text Extraction]
    C --> E[Layout Analysis]
    D --> F[Japanese Text Normalization]
    F --> G[Field Extraction]
    E --> G
    G --> H[Structured Data]
    
    subgraph "Japanese Document Processing"
        B
        direction TB
        B1[Deskew] --> B2[Noise Reduction]
        B2 --> B3[Contrast Enhancement]
        B3 --> B4[Orientation Detection]
    end
```

## Optimization Process

```mermaid
flowchart LR
    A[Base Models] --> B{Optimization Algorithm}
    B -->|Genetic| C1[Crossover & Mutation]
    B -->|BMR| C2[Best-Mean-Random]
    B -->|BWR| C3[Best-Worst-Random]
    C1 --> D[Model Merging]
    C2 --> D
    C3 --> D
    D --> E[Merged Model]
    E --> F[Evaluation]
    F --> G{Criteria Met?}
    G -->|No| B
    G -->|Yes| H[Final Optimized Model]
```

## Complete System Workflow

```mermaid
sequenceDiagram
    participant User
    participant CLI as Command Line Interface
    participant Download as Model Download
    participant Data as Dataset Processing
    participant Optimize as Optimization Engine
    participant Eval as Evaluation System
    participant Demo as Interactive Demo
    
    User->>CLI: run_complete_workflow.py
    CLI->>Download: setup_resources.py
    Download->>Download: Download models from HuggingFace
    Download-->>CLI: Models downloaded
    
    CLI->>Data: Prepare evaluation data
    Data-->>CLI: Data prepared
    
    CLI->>Optimize: Run optimization algorithms
    
    Optimize->>Optimize: Run genetic algorithm
    Optimize->>Optimize: Run BMR algorithm
    Optimize->>Optimize: Run BWR algorithm
    
    Optimize-->>CLI: Optimization complete
    
    CLI->>Eval: Compare algorithm results
    Eval-->>CLI: Best algorithm identified
    
    CLI->>Demo: Setup best model for demo
    Demo-->>CLI: Demo prepared
    
    CLI-->>User: Workflow complete
    
    User->>Demo: Test best model
    Demo-->>User: OCR results
```

## Component Structure

```mermaid
graph TD
    subgraph "Project Structure"
        CONFIG[configs/]
        EVOMERGE[evomerge/]
        ASSETS[assets/]
        DATA[data/]
        SCRIPTS[Scripts]
    end
    
    subgraph "Config Files"
        CONFIG --> CONFIG_EVOLUTION[evolution/]
        CONFIG --> CONFIG_LLM[llm/]
        CONFIG --> CONFIG_VLM[vlm/]
        
        CONFIG_EVOLUTION --> EVOLUTION_DEFAULT[default.yaml]
        CONFIG_EVOLUTION --> EVOLUTION_BMR_BWR[bmr_bwr_config.yaml]
    end
    
    subgraph "Core Components"
        EVOMERGE --> EVOMERGE_DATA[data/]
        EVOMERGE --> EVOMERGE_EVAL[eval/]
        EVOMERGE --> EVOMERGE_EVOLUTION[evolution/]
        EVOMERGE --> EVOMERGE_MODELS[models/]
        EVOMERGE --> EVOMERGE_PROCESSORS[processors/]
        
        EVOMERGE_DATA --> DATA_DATASETS[datasets.py]
        EVOMERGE_DATA --> DATA_PROCESSOR[processor.py]
        
        EVOMERGE_EVOLUTION --> EVOLUTION_GENETIC[genetic.py]
        EVOMERGE_EVOLUTION --> EVOLUTION_BMR[bmr.py]
        EVOMERGE_EVOLUTION --> EVOLUTION_BWR[bwr.py]
        
        EVOMERGE_MODELS --> MODELS_BASE[base_ocr.py]
        EVOMERGE_MODELS --> MODELS_JAPANESE[japanese_ocr.py]
        EVOMERGE_MODELS --> MODELS_FIELD[field_extractor.py]
    end
    
    subgraph "Scripts"
        SCRIPTS --> SCRIPT_DEMO[demo.py]
        SCRIPTS --> SCRIPT_EVALUATE[evaluate.py]
        SCRIPTS --> SCRIPT_OPTIMIZE[run_optimization.py]
        SCRIPTS --> SCRIPT_WORKFLOW[run_complete_workflow.py]
        SCRIPTS --> SCRIPT_SETUP[setup_resources.py]
        SCRIPTS --> SCRIPT_COMPARE[compare_results.py]
    end
    
    subgraph "Data"
        DATA --> DATA_SAMPLE[sample_config.yaml]
    end
```

## Optimization Algorithm Comparison

```mermaid
graph LR
    subgraph "Genetic Algorithm"
        GA1[Selection] --> GA2[Crossover]
        GA2 --> GA3[Mutation]
        GA3 --> GA4[Evaluation]
    end
    
    subgraph "BMR Algorithm"
        BMR1[Get Best Solution] --> BMR2[Calculate Mean]
        BMR2 --> BMR3[Select Random Solution]
        BMR3 --> BMR4[Apply Formula]
        BMR4 --> BMR5[Random Exploration]
    end
    
    subgraph "BWR Algorithm"
        BWR1[Get Best Solution] --> BWR2[Get Worst Solution]
        BWR2 --> BWR3[Select Random Solution]
        BWR3 --> BWR4[Apply Formula]
        BWR4 --> BWR5[Random Exploration]
    end
```

## BMR Algorithm Workflow

```mermaid
flowchart TD
    START[Start BMR Algorithm] --> INIT[Initialize Population]
    INIT --> EVAL[Evaluate Fitness]
    EVAL --> BEST[Identify Best Individual]
    BEST --> MEAN[Calculate Mean of Population]
    MEAN --> LOOP{For Each Individual}
    LOOP --> RANDOM[Pick Random Individual]
    RANDOM --> PROB{Random >= 0.5?}
    
    PROB -->|Yes| EXPLOIT[Apply BMR Formula:<br>V' = V + r1(Best - T×Mean) + r2(Best - Random)]
    PROB -->|No| EXPLORE[Apply Random Exploration:<br>V' = Upper - (Upper-Lower)×r3]
    
    EXPLOIT --> NEXT[Next Individual]
    EXPLORE --> NEXT
    NEXT --> LOOP_END{End of Population?}
    LOOP_END -->|No| LOOP
    LOOP_END -->|Yes| ELITISM[Preserve Elite Individuals]
    ELITISM --> NEXT_GEN[Next Generation]
    NEXT_GEN --> TERM{Termination?}
    TERM -->|No| EVAL
    TERM -->|Yes| END[Return Best Solution]
```

## BWR Algorithm Workflow

```mermaid
flowchart TD
    START[Start BWR Algorithm] --> INIT[Initialize Population]
    INIT --> EVAL[Evaluate Fitness]
    EVAL --> BEST[Identify Best Individual]
    BEST --> WORST[Identify Worst Individual]
    WORST --> LOOP{For Each Individual}
    LOOP --> RANDOM[Pick Random Individual]
    RANDOM --> PROB{Random >= 0.5?}
    
    PROB -->|Yes| EXPLOIT[Apply BWR Formula:<br>V' = V + r1(Best - T×Random) - r2(Worst - Random)]
    PROB -->|No| EXPLORE[Apply Random Exploration:<br>V' = Upper - (Upper-Lower)×r3]
    
    EXPLOIT --> NEXT[Next Individual]
    EXPLORE --> NEXT
    NEXT --> LOOP_END{End of Population?}
    LOOP_END -->|No| LOOP
    LOOP_END -->|Yes| ELITISM[Preserve Elite Individuals]
    ELITISM --> NEXT_GEN[Next Generation]
    NEXT_GEN --> TERM{Termination?}
    TERM -->|No| EVAL
    TERM -->|Yes| END[Return Best Solution]
```

## Japanese OCR Pipeline

```mermaid
flowchart TD
    START[Japanese Invoice/Receipt] --> PREPROC[Image Preprocessing]
    PREPROC --> OCR[OCR Model]
    OCR --> TEXT[Raw Text Extraction]
    TEXT --> NORM[Japanese Text Normalization]
    NORM --> FIELD[Field Extraction]
    FIELD --> STRUCT[Structured Data]