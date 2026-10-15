# Event-Driven-Brain-Network-Dynamics
This repository contains Python scripts for characterizing developmental and intrinsic functional brain network dynamics using an event-based community evolution framework.
Overview
Four types of community evolution events are considered: Emergence, Disappearance, Expansion, and Contraction.
Event-specific co-occurrence connectivity (COC) is used to characterize the coordinated participation of brain regions in these events. Cross-timescale alignment (CTA) quantifies the correspondence between developmental and intrinsic COC patterns.
Code Description
01_developmental_COC.py
Computes functional connectivity matrices, performs community detection, identifies community evolution events, and derives developmental COC.
02_intrinsic_COC.py
Identifies community evolution events from individual resting-state fMRI time windows. Intrinsic COC is calculated using the same event-specific COC definitions as the developmental analysis.
03_CTA_calculation.py
Quantifies the spatial correspondence between developmental and individual intrinsic COC patterns.
04_COC_age_associations.py
Examines partial correlations between intrinsic COC and gestational age (GA) and postnatal age (PNA).
Requirements
Python and the dependencies used by the scripts are required. See requirements.txt for details.
Data Availability
Neuroimaging data are not included in this repository. Data access is subject to the relevant dataset provider's terms.
Citation
Citation information will be updated following publication.
