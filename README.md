# EEGNiPrep (working name)

NiPreps-style EEG BIDS-App scaffold inspired by fMRIPrep/PETPrep contracts.

## Why EEGNiPrep?
The `eegprep` name is already in use in other EEG projects. This repository adopts
**EEGNiPrep** as a differentiating working name while keeping the executable command
`eegprep` for now.

## Current status
This repository now contains:
- A BIDS-App style CLI signature.
- Workflow factories (`init_eegprep_wf`, `init_single_subject_wf`).
- Placeholder QC/preprocessing/output modules matching the target architecture.
- Stub derivative metrics JSON writing under `sub-<id>/eeg/`.

## Example command
```bash
eegprep /path/to/bids /path/to/derivatives participant \
  --participant-label 01 02 \
  --task rest \
  --session-label 01 \
  --work-dir /path/to/work \
  --nprocs 8 \
  --omp-nthreads 2 \
  --level minimal \
  --eeg-reference average \
  --montage auto \
  --high-pass 0.3 \
  --notch auto \
  --source-recon none
```
