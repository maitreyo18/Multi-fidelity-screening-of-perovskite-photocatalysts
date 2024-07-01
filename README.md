In this project we have showcased the success of surrogate Machine Learning (ML) models trained on high-throughput density functional theory (HT-DFT) data, to identify Halide Perovskites (HaPs) for efficient photocatalytic water splitting. We trained our ML model on a multi-fidelity HT-DFT dataset of total 985 points (614 PBE + 371 HSE) comprising the target properties of Decomposition Energy and Band Gap. We implemented the state-of-the-art Regularized Greey Forest (RGF) (https://github.com/RGF-team/rgf) regression to generate the predictive models for Decomposition Energy and Band gap. Next, we enumerated a dataset of 151,140 hypothetical perovskites in different phases and used the surrogate models to screen and identify through the huge combinatorial alloyed HaP space. All the python scripts used to perform the ensemble of 4000 RGF models and to calculate the theoretical Solar-to-Hydrogen (STH) efficiency, have been provided in the "./codes" directory. All the ML predictions have been provided in the .csv files:

[1] Training_PBE_and_HSE_data.csv : The multi0fidelity training dataset used to train the surrogate models.

[2] Tol_screened_ensemble_final.csv : ML predicted Decomposition Energy (PBE and HSE) and Band Gap (HSE) of the 67,916 perovskites that complied with the tolerance factors out of the total 151,140 perovskites over 4000 models.

[3] 3043_screened_comps.csv : List of screened perovskites identified using the hierarchical screening funnel suitable for photocatalytic water splitting.

[4] 1173_Pb-free_comps.csv : List of Pb-free screened perovskite photocatalysts.

[5] DFT_validated_perovskites.xlsx : DFT validation of 5 chosen perovskites against the ML predictions.

"./PBE-relaxed_cif_files" contain the PBE-relaxed .cif files of the DFT validated perovskites. 
