# About This Project

## Overview

This project combines deep learning and 3D reconstruction to model the lamina cribrosa (LC), a critical ocular structure, from optical coherence tomography (OCT) imagery. The lamina cribrosa is a porous connective tissue located at the optic nerve head, and its morphology is a key biomarker in glaucoma progression and other ophthalmic pathologies. By automating pore segmentation in 2D OCT slices and stacking them into a volumetric representation, we can compute quantitative 3D biomarkers that would otherwise require manual, time-consuming analysis.

## Technical Approach

The pipeline is built in three stages:

**Stage 1: 2D Segmentation** uses an Attention U-Net (a PyTorch-based encoder-decoder architecture with attention gates on skip connections) to segment pores in individual OCT slices at 256×256 resolution. The model is trained on paired OCT images and hand-annotated pore masks using a combined Dice + weighted BCE loss, with class weighting (`pos_weight=5`) to handle the severe class imbalance (pores are sparse relative to background connective tissue). Critically, the training/validation split is **patient-based** — entire patients are assigned to either train or val — to avoid data leakage and ensure honest evaluation.

**Stage 2: Single-Patient Inference** loads the trained model and runs it on all OCT slices from a single patient (e.g., 69 consecutive slices), producing a per-slice binary mask for each. Masks are saved as PNG files, preserving the original slice dimensions.

**Stage 3: 3D Reconstruction & Biomarkers** stacks the 2D masks into a 3D volume (Z, H, W) and computes three quantitative biomarkers:
- **CTVF** (connective tissue volume fraction): the ratio of non-pore voxels to total voxels, indicating how much of the LC is solid tissue
- **3D pore count**: the number of distinct connected components in the volume using 26-connectivity
- **Mean pore volume**: the average size (in voxels) of each pore, a proxy for pore uniformity

The code also renders a 3D voxel plot of the connective tissue beams for visual inspection.

## Data & Training

The dataset comprises OCT images from multiple patients, each with manually annotated pore masks. The pipeline is designed to work with nested folder structures (patient → eye → laterality → slices) and automatically discovers and pairs image-mask files based on matching filenames and relative paths. With ~10 patients and ~2,000 total image-mask pairs, the patient-based split yields roughly 8 patients (1,850 images) for training and 2 patients (470 images) for validation.

Training is fast on a single GPU (T4 or better): 25 epochs over ~2,000 images at 256×256 takes 45–75 minutes on Colab's free-tier GPU. The best model (by validation IoU) is automatically checkpointed and persisted to Google Drive, so training can be resumed or interrupted without loss of progress.

## Applications

This work enables:
- **Automated morphometric analysis** of the lamina cribrosa without manual segmentation
- **Longitudinal studies** tracking LC biomarkers over time in glaucoma patients
- **Clinical stratification** based on 3D pore morphology and connective tissue density
- **Hypothesis generation** about structure-function relationships in optic nerve disease

By combining the robustness of deep learning with the spatial fidelity of 3D reconstruction, the pipeline bridges the gap between 2D image analysis and clinically actionable 3D biomarkers.

## Key Design Decisions

1. **Attention U-Net over standard U-Net:** attention gates allow the decoder to focus on task-relevant features from the encoder, improving segmentation accuracy on sparse, small objects (pores).
2. **Patient-based validation split:** prevents the model from inflating its score by memorizing adjacent slices from the same patient.
3. **26-connectivity for 3D pore labeling:** allows pores separated by a single voxel diagonal or face-to-face connection to be counted as distinct, better capturing the LC's actual morphology.
4. **Weighted BCE + Dice loss:** BCE with `pos_weight=5` penalizes false negatives on the minority class (pores), while soft Dice provides a differentiable approximation of the Dice coefficient, a standard segmentation metric.
5. **Colab deployment:** free GPU access, persistent Drive storage, and minimal setup overhead make the pipeline accessible to researchers without local computing infrastructure.

## Future Directions

- Fine-tuning on domain-specific datasets (e.g., glaucoma vs. healthy controls)
- Uncertainty quantification (Bayesian approaches or ensemble methods) to flag low-confidence predictions
- Transfer learning from larger medical imaging datasets
- Multi-scale analysis (e.g., segmenting vessels, microglial cells, or other LC substructures)
- Integration with functional OCT metrics (blood flow, structural changes) for structure-function analysis
