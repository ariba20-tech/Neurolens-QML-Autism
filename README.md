NeuroLens: Explainable Quantum Machine Learning for Autism Spectrum Analysis

Live Application:
👉 https://neurolens-qml-autism.vercel.app/

🌌 Overview

NeuroLens is an experimental Explainable Quantum Machine Learning (QML) platform designed to analyze behavioral and clinical indicators associated with Autism Spectrum Disorder (ASD).

The project demonstrates how hybrid quantum-classical AI systems can transform traditional black-box machine learning models into glass-box systems, enabling transparent and interpretable decision making.

Instead of producing predictions without explanation, NeuroLens exposes feature influence, probability contributions, and interpretable outputs, making AI more trustworthy for healthcare research.

This project combines:

Quantum Machine Learning

Explainable AI

Neurodevelopmental data analysis

Interactive visualization

The system is deployed as a web application for demonstration and educational purposes.

🧠 What is Autism Spectrum Disorder (ASD)?

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition characterized by differences in:

Social communication

Behavioral patterns

Sensory processing

Cognitive interactions

ASD presents differently in every individual, which is why it is called a spectrum disorder.

Diagnosing autism is complex because it relies primarily on behavioral observation and clinical assessments rather than a single definitive medical test.

Researchers have increasingly explored machine learning and AI techniques to assist clinicians in analyzing behavioral and neurological patterns associated with autism.

However, most AI systems currently used in healthcare suffer from a major limitation:

They behave as black-box models, producing predictions without revealing how those predictions were made.

⚠️ The Black-Box Problem in AI

A black-box model is an AI system whose internal reasoning is hidden.

Example:

Input → Deep Neural Network → Output

The model produces a prediction but does not explain why.

In healthcare, this is problematic because:

Clinicians must justify decisions

Predictions affect patient care

Ethical accountability is required

Therefore, modern research focuses on Explainable AI (XAI) systems that transform black boxes into transparent models.

🔎 Glass-Box AI: Transparent Machine Learning

A glass-box model is an AI system whose internal reasoning can be understood.

Instead of hidden computations, a glass-box model exposes:

Feature influence

Decision pathways

Probability contributions

In NeuroLens, this is achieved through quantum measurement analysis and feature importance visualization.

⚛️ What is Quantum Machine Learning (QML)?

Quantum Machine Learning (QML) is an emerging field that combines quantum computing principles with machine learning algorithms.

Quantum computing introduces concepts such as:

Superposition

A quantum bit (qubit) can represent multiple states simultaneously.

Entanglement

Quantum states can become correlated, enabling complex relationships between variables.

Quantum Interference

Quantum amplitudes can reinforce or cancel each other, allowing sophisticated pattern recognition.

QML models leverage quantum circuits and variational algorithms to learn patterns in data using hybrid classical-quantum architectures.

These models are particularly promising for:

high-dimensional datasets

nonlinear relationships

complex pattern discovery

Recent research demonstrates that QML systems can improve classification performance in medical datasets compared to classical models.

🚀 Why Quantum ML for Autism Analysis?

Autism datasets often contain:

heterogeneous behavioral features

subtle correlations

nonlinear relationships

high-dimensional patterns

Traditional models sometimes struggle to capture these relationships.

Quantum models allow:

richer feature representation

high-dimensional embeddings

improved pattern detection

This makes QML a promising approach for neurodevelopmental data analysis.

🧩 System Architecture

The NeuroLens platform follows a hybrid quantum-classical pipeline.

User Input
   ↓
Feature Processing
   ↓
Quantum Feature Encoding
   ↓
Variational Quantum Circuit
   ↓
Measurement Probabilities
   ↓
Prediction
   ↓
Explainability Engine
   ↓
Interactive Visualization
Components
1️⃣ Data Processing

Autism-related behavioral features are normalized and converted into feature vectors.

2️⃣ Quantum Feature Encoding

Classical features are embedded into quantum states.

3️⃣ Variational Quantum Circuit

Parameterized quantum circuits perform the learning process.

4️⃣ Measurement Layer

Quantum measurements generate probability outputs.

5️⃣ Explainability Engine

Feature importance and decision explanations are computed.

🧪 Explainability Mechanism

To overcome the black-box problem, NeuroLens introduces a glass-box explainability layer.

This includes:

Feature Importance Mapping

Determines which behavioral features influence the model most.

Example:

Eye Contact Score      → 0.41
Communication Ability  → 0.32
Attention Level        → 0.19
Repetitive Behavior    → 0.08
Probability Attribution

Each prediction includes probability contributions derived from quantum measurements.

Sensitivity Analysis

The model observes how prediction probabilities change when features are perturbed.

These techniques create transparent and interpretable AI outputs.

🌐 Rendered Application

The system is deployed at:

👉 https://neurolens-qml-autism.vercel.app/

The web interface allows users to:

input behavioral indicators

run the QML model

visualize predictions

inspect feature influence

The UI focuses on educational explainability rather than clinical diagnosis.

⚙️ Tech Stack
Frontend

Next.js

React

TailwindCSS

Chart.js

Backend

Python

FastAPI

Machine Learning

Scikit-learn

NumPy

Pandas

Quantum Computing

PennyLane

Qiskit

Visualization

Plotly

Matplotlib

Deployment

Vercel

📊 Example Prediction Flow
User Inputs Behavioral Features
        ↓
Feature Vector Created
        ↓
Encoded into Quantum Circuit
        ↓
Variational Quantum Model Runs
        ↓
Measurement Probabilities Calculated
        ↓
Prediction Generated
        ↓
Explainability Dashboard Displays Results
🎯 Key Features

✔ Hybrid Quantum-Classical Machine Learning
✔ Explainable AI for healthcare research
✔ Feature importance visualization
✔ Interactive web interface
✔ Quantum measurement-based predictions

⚠️ Ethical Disclaimer

This system is not a medical diagnostic tool.

It is intended for:

educational purposes

research demonstrations

explainable AI experimentation

Clinical diagnosis of autism should always be conducted by qualified medical professionals.

🔮 Future Improvements

Planned enhancements include:

Multimodal datasets (EEG, eye-tracking, speech)

Real quantum hardware integration

Larger quantum circuits

Clinical decision-support dashboards

📚 References

Applications of machine learning for autism diagnosis

Quantum Machine Learning fundamentals

Hybrid QML frameworks for medical analysis

👩‍💻 Author

Ariba Hussain
AIML Student | Quantum AI Research Enthusiast
