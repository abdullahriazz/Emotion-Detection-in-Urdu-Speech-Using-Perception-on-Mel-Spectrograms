Urdu Speech Emotion Recognition via Visual PerceptionThis repository contains the implementation for a comparative study on Urdu Speech Emotion Recognition (SER), benchmarking Audio Spectrogram Transformers (AST) against a CNN baseline (PANNs). The approach treats SER as a computer vision task by processing Log-Mel Spectrograms.Project OverviewObjective: Adapt Vision Transformers for low-resource language SER.Dataset: UrduSER (Naturalistic speech from Pakistani drama serials).Architecture: Comparative analysis between inductive bias (CNN) and global attention (Transformer).Key Result: The AST architecture achieves a 10% improvement over the CNN baseline (75% vs. 65% Accuracy), validating the efficacy of self-attention for capturing Urdu prosody.DatasetThe project utilizes the UrduSER dataset (Version 3/4).Source: Naturalistic audio from TV talk shows and dramas.Classes: Consolidated to 4 primary emotions: Angry, Happy, Neutral, Sad.Sample Size: ~2,400 samples (Balanced).Splitting Strategy: Speaker-Independent Hold-out (80/20). The test set consists entirely of speakers not seen during training to ensure generalization.Methodology1. PreprocessingAudio is converted to visual representations using the following parameters:Sample Rate: 32 kHzRepresentation: Log-Mel SpectrogramMel Bins: 64Window Size: 1024Hop Length: 320Duration: Fixed 5-second context (padded/truncated).2. ArchitecturesAST (Audio Spectrogram Transformer): Vision Transformer backbone pre-trained on ImageNet and AudioSet. Processes spectrograms as sequences of $16 \times 16$ patches.PANNs (Cnn14): 14-layer CNN pre-trained on AudioSet. Serves as the standard inductive bias baseline.3. Training StrategyTo prevent catastrophic forgetting of pre-trained weights, a Two-Stage Finetuning Protocol is implemented:Stage 1: Backbone frozen; Linear classification head trained.Stage 2: End-to-end finetuning with differential learning rates (Lower LR for backbone, higher LR for head).InstallationPrerequisites: Python 3.8+, GPU with 12GB+ VRAM.Note: The panns-inference library has a conflict with newer protobuf versions. You must install protobuf<4.21.0.Bash# 1. Install PyTorch (Ensure CUDA version matches your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Install PANNs with Protobuf fix
pip install panns-inference
pip install "protobuf<4.21.0"

# 3. Install remaining dependencies
pip install timm pandas scikit-learn matplotlib seaborn tqdm librosa
UsageData SetupOrganize the dataset in the input directory. Labels are inferred from directory names.input/
└── urdu_dataset/
    ├── Angry/
    ├── Happy/
    ├── Neutral/
    └── Sad/
ConfigurationAdjust hyperparameters in src/config.py:Pythonclass Config:
    SR = 32000              # Fixed for PANNs compatibility
    BATCH_SIZE = 8          # Adjust based on VRAM
    EPOCHS = 30
    LR_BACKBONE = 1e-5      # Stage 2 Backbone LR
    LR_HEAD = 3e-4          # Stage 1/2 Head LR
    
TrainingExecute the training script. The pipeline will:Initialize the Speaker-Independent split.Download pre-trained weights.Execute the Two-Stage training loop.Save the model with the lowest validation loss.ResultsPerformance metrics on the held-out test set:ModelAccuracyWeighted F1Angry (F1)Happy (F1)PANNs (CNN)65%0.630.760.47AST (Transformer)75%0.750.850.64Both models exhibit confusion between Happy and Neutral classes, attributed to the subtle acoustic markers of positive valence in the naturalistic dataset.ReferencesRiaz, A. & Lodhi, Z. K. Emotion Detection in Urdu Speech Using Perception on Mel Spectrograms. Technical Report.Akhtar, M. Z., et al. "UrduSER: A Dataset for Urdu Speech Emotion Recognition". Mendeley Data, V3, 2024.Gong, Y., et al. "AST: Audio Spectrogram Transformer". Interspeech, 2021.Kong, Q., et al. "PANNs: Large-Scale Pretrained Audio Neural Networks". IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2020.
