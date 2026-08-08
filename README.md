```mermaid
graph TD
    subgraph Offline["Offline Training & Export"]
        A[BCI IV-2a Dataset<br/>via MOABB] --> B[EEGNet Training<br/>PyTorch + MPS]
        B --> C[Per-Subject Checkpoints<br/>.pt]
        C --> D[ONNX Export<br/>+ Numerical Verification]
        D --> E[Subject Model<br/>.onnx]
    end

    subgraph Streaming["Real-Time Simulation"]
        F[Continuous Raw EEG<br/>+ Event Markers] --> G[LSL Outlet<br/>lsl_outlet.py]
        G -->|EEG Stream| H((LSL Network))
        G -->|Marker Stream| H
    end

    subgraph App["Unified Streamlit App"]
        H --> I[Rolling Buffer<br/>Background Thread]
        I --> J[Bandpass Filter<br/>4-38Hz, Leading Margin]
        E --> K[ONNX Runtime<br/>Inference]
        J --> K
        K --> L[Prediction Smoothing<br/>+ Confidence Gating]
        L --> M[Live Monitor Page<br/>Intent · Command · Analytics]
        F -.recorded session.-> N[Forensics Replay Page<br/>Scrubbable Analysis]
        E -.-> N
    end

    style Offline fill:#12151C,stroke:#1F232D,color:#E9EBF0
    style Streaming fill:#12151C,stroke:#1F232D,color:#E9EBF0
    style App fill:#12151C,stroke:#1F232D,color:#E9EBF0
```