# qsiprep-coreg-check
Apptainer recipe for python-based dwiref-T1w coregistration check

## Build

```sh
apptainer build coreg.sif coreg.def
```

## Usage

```sh
apptainer run --no-home --cleanenv --bind /path/to/bids/derivatives/qsiprep/${subject}:/data \
/path/to/coreg.sif /data/anat/${subject}_space-ACPC_desc-preproc_T1w.nii.gz \
/data/${session}/dwi/${subject}_${session}_space-ACPC_dwiref.nii.gz \
/data/${session}/figures/${subject}_${session}_space-ACPC_desc-dwirefT1wCoReg.html
```
