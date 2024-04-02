### Python Source Code for master's thesis

Requisite modules: `numpy`, `matplotlib`, `pandas`, `seaborn`, `scipy`, `h5py`
## Note: only works with access to IEMR OUS R-disk; access to TPM combodata. 

### `CombodataSR_2D.py`
Contains a python class that takes combodata (with TPM magnitude, velocity fields in two spatial dimensions + time) as input to analyze 2D LV deformation. It produces and returns global strain rate data and can save these as separate files, as well as save videos that visualize the velocity vector field and strain rate as ellipses. See bottom of file for example of use.

### `CombodataSR_3D.py`
Contains a python class that takes stacks of combodata (with TPM magnitude, velocity fields in two spatial dimensions + time) as input to analyze 3D LV deformation. It produces and returns global strain rate data and can save these as separate files, as well as save videos that visualize the velocity vector field and strain rate as ellipses. See bottom of file for example of use.

### `util.py`
Collection of functions that are imported in other scripts.

### `strain_analysis.py`
Script that uses `CombodataSR_2D.py` to collect data from all combodata files, and organizes them in a pandas dataframe. The dataframe can be found in this folder saved as `combodata_analysis`, and can be imported in the script to plot linear regressions and other statistical analysis.

### `strain_analysis_3d.py`
Script that uses `CombodataSR_3D.py` to collect data from all combodata stacks, and organizes them in a pandas dataframe. The dataframe can be found in this folder saved as `combodata_analysis_3d`, and can be imported in the script to plot linear regressions and other statistical analysis.

### `combodata_ellipse_3d.py`
Input a combodata stack covering a LV to quality check the segmentation and order of the slices, and produce plots that visualize regional variation between slices.
