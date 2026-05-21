
Repository for publication "Meshless methods application for mean field game coefficient inverse problem"

# Collocation method

`script.py` - main file with computation of all experiments, resulting errors, their mean and standart deviations.

`settings/simplest_mfg.yaml` - settings for the spline approximator of the direct problem, used for generation of measurements data.

`colloc_solutoin_coefs.pkl` - saved coefficients for the spline approximation of the direct problem.

`settings/simplest_mfg_inverse.yaml` - settings for the spline approximator of the inverse problem.


# Requirements

pyyaml
numpy
pickle
sib-pinn (see github.com/ANever/sib-pinn)