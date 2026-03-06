# Failure-Aware ACT
### Recoverability-Guided Chunk Intervention for Visuomotor Imitation Learning

This project introduces a **failure-aware execution layer for Action Chunking Transformers (ACT)** that enables robots to detect impending task failure during execution and **intervene early enough to recover**.

Instead of treating failures as something to fix during training (e.g., DAgger-style dataset aggregation), this system allows the robot to **monitor its own trajectory at runtime**, detect when its predicted action chunk is becoming invalid, and **interrupt and replan before crossing a point of no return**.

The result is a robot policy that can **notice drift, judge recoverability, and salvage execution in real time.**

---

# Overview

Visuomotor imitation learning policies like **ACT** generate **multi-step action chunks** from demonstrations. While chunking improves smoothness and reduces compounding error, it introduces a new problem:

**Once a chunk is predicted, the robot commits to it even if the world changes.**

Small perturbations such as:

- object shifts  
- grasp misalignment  
- sensor noise  
- occlusion  
- actuation error  

can cause the predicted chunk to become invalid before it finishes executing.

This project introduces a **runtime monitoring and intervention layer** that:

1. Detects when the current action chunk is drifting toward failure  
2. Estimates whether the task is still recoverable  
3. Interrupts execution at the optimal moment  
4. Replans using ACT from the updated state  

---

# Key Idea

The system converts ACT from a passive imitation policy into a **failure-aware control policy**.

Instead of asking:

> Did the robot fail?

it asks:

> Is the **current predicted chunk** still valid?

and

> If we interrupt now, can the task still be recovered?

---

# System Architecture

```
        Observations
             │
             ▼
     ┌─────────────────┐
     │   ACT Policy    │
     │ (Chunked Policy)│
     └─────────────────┘
             │
     Predicted Action Chunk
             │
             ▼
     ┌─────────────────────────┐
     │ Failure Monitoring Head │
     │                         │
     │ predicts:               │
     │  - failure risk         │
     │  - recoverability       │
     │  - intervention score   │
     └─────────────────────────┘
             │
             ▼
      Intervention Logic
             │
   ┌─────────┴─────────┐
   │                   │
Continue Chunk    Interrupt + Replan
   │                   │
   ▼                   ▼
Execute action    Query ACT again
```

---

# Core Components

## 1. ACT Policy

The base controller is **Action Chunking Transformer (ACT)**.

ACT predicts a sequence of actions:

```
a_t, a_{t+1}, ... a_{t+H}
```

rather than single-step actions.

Inputs:
- RGB observations
- proprioception
- action history

Outputs:
- future action chunk

---

## 2. Failure Monitoring Head

A lightweight network that predicts:

```
failure_risk(s_t)
recoverability(s_t)
intervention_score(s_t)
```

Inputs include:

- observation embeddings
- ACT latent features
- predicted action chunk
- chunk entropy / uncertainty
- observation drift signals

---

## 3. Recoverability Estimation

Recoverability measures whether a state is **before or after the point of no return**.

Definition:

```
recoverability(s_t) =
  probability that replanning from s_t still succeeds
```

This is estimated using:

- simulated replanning rollouts
- trajectory success labels
- learned prediction heads

---

## 4. Chunk Interruption Policy

At each step:

```
if risk < threshold:
    continue

elif risk high and recoverable:
    interrupt and replan

else:
    reset / backtrack
```

This allows the robot to **cut a chunk mid-execution**.

---

# Why This Is Novel

Existing work focuses on **detecting failures**.

Examples include:
- runtime OOD detection
- uncertainty estimation
- failure classification

However, most systems **only raise alarms**.

This project introduces **failure-aware chunk control**, which includes:

- predicting when the current ACT chunk becomes invalid  
- estimating whether the trajectory is still recoverable  
- selecting an optimal interruption point  
- using corrective continuation rather than stopping  

In other words:

**detect → intervene → recover**

instead of

**detect → stop**

---

# Project Goals

The system should:

- improve robustness of imitation policies
- recover from perturbations during execution
- detect the point of no return
- improve task success under distribution shift

---

# Metrics

Evaluation focuses on **task outcomes**, not only detection accuracy.

Primary metrics:

```
Task Success Rate
Recovered Success Rate
Failure Prediction Lead Time
False Intervention Rate
Average Interventions per Episode
```

Secondary metrics:

```
Failure Detection AUROC
Recoverability Prediction Error
```

---

# Repository Structure

```
failure-aware-act/
│
├── act/
│   ├── model.py
│   ├── train_act.py
│   └── dataset.py
│
├── monitor/
│   ├── risk_head.py
│   ├── recoverability_head.py
│   └── monitor_model.py
│
├── rollout/
│   ├── intervention_policy.py
│   ├── rollout_runner.py
│   └── perturbations.py
│
├── simulation/
│   ├── env_setup.py
│   └── task_configs.py
│
├── evaluation/
│   ├── metrics.py
│   └── experiment_runner.py
│
└── notebooks/
    ├── training.ipynb
    └── analysis.ipynb
```

---

# Training Pipeline

## Step 1 — Train Baseline ACT

Train ACT on demonstration data.

```bash
python train_act.py
```

---

## Step 2 — Generate Rollouts

Run ACT in simulation with perturbations.

```bash
python rollout_runner.py
```

Logs include:

- observations
- action chunks
- latent features
- outcomes

---

## Step 3 — Train Failure Monitor

Train monitoring head using logged trajectories.

```bash
python train_monitor.py
```

Targets:

```
failure_within_k_steps
recoverability_score
```

---

## Step 4 — Enable Runtime Intervention

Run ACT with failure-aware control enabled.

```bash
python run_with_intervention.py
```

---

# Simulation Experiments

Recommended tasks:

- Pick and place
- Grasp correction
- Peg insertion
- Drawer opening

Perturbations:

- object pose shift
- occlusion
- actuation noise
- sensor delay

---

# Hardware Experiments (Optional)

The system can also be tested on a physical robot.

Example tasks:

- grasp recovery
- object alignment
- pick-and-place under perturbation

Execution loop:

```
ACT → monitor → intervention → replanning
```

---

# Installation

```bash
git clone https://github.com/virkvarjun/Research.git

cd Research

pip install -r requirements.txt
```

---

# Dependencies

```
PyTorch
Transformers
Robotics simulator (Isaac / PyBullet / MuJoCo)
NumPy
OpenCV
```

---

# Future Extensions

Possible upgrades include:

- short-horizon world models for failure prediction
- conformal risk calibration
- multi-task VLA integration
- semantic failure classification using vision-language models

---

# Citation

If you use this project, please cite:

```
Failure-Aware ACT:
Recoverability-Guided Chunk Intervention
for Visuomotor Imitation Learning
```

---

# Author

**Arjun Virk**  
University of Waterloo  
AI & Robotics Research
