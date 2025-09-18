import pyhrv
import pyhrv.tools as tools
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cbook
import matplotlib as mpl
from matplotlib.projections import register_projection

from matplotlib.ticker import FixedLocator, LogFormatter, ScalarFormatter
from matplotlib.scale import FuncScale

def radar_chart(nni=None,
                rpeaks=None,
                comparison_nni=None,
                comparison_rpeaks=None,
                parameters=None,
                reference_label='Reference',
                comparison_label='Comparison',
                save_path=None,
                legend=True):
    """Plots a radar chart of HRV parameters to visualize the evolution the parameters computed from a NNI series
    (e.g. extracted from an ECG recording while doing sports) compared to a reference/baseline NNI series (
    e.g. extracted from an ECG recording while at rest).

    The radarchart normalizes the values of the reference NNI series with the values extracted from the baseline NNI
    series being used as the 100% reference values.

    Example:    Reference NNI series:    SDNN = 100ms → 100%
                Comparison NNI series:    SDNN = 150ms → 150%

    The radar chart is not limited by the number of HRV parameters to be included in the chart; it dynamically
    adjusts itself to the number of compared parameters.

    Docs: https://pyhrv.readthedocs.io/en/latest/_pages/api/tools.html#radar-chart-radar-chart

    Parameters
    ----------
    nni : array
        Baseline or reference NNI series in [ms] or [s] (default: None)
    rpeaks : array
        Baseline or referene R-peak series in [ms] or [s] (default: None)
    comparison_nni : array
        Comparison NNI series in [ms] or [s] (default: None)
    comparison_rpeaks : array
        Comparison R-peak series in [ms] or [s] (default: None)
    parameters : list
        List of pyHRV parameters (see keys of the hrv_keys.json file for a full list of available parameters).
        The list must contain more than 1 pyHRV parameters (default: None)
    reference_label : str, optional
        Plot label of the reference input data (e.g. 'ECG while at rest'; default: 'Reference')
    comparison_label : str, optional
        Plot label of the comparison input data (e.g. 'ECG while running'; default: 'Comparison')
    show : bool, optional
        If True, shows plot figure (default: True).
    legend : bool, optional
        If true, add a legend with the computed results to the plot (default: True)

    Returns (biosppy.utils.ReturnTuple Object)
    ------------------------------------------
    [key : format]
        Description.
    reference_results : dict
        Results of the computed HRV parameters of the reference NNI series
        Keys:    parameters listed in the input parameter 'parameters'
    comparison results : dict
        Results of the computed HRV parameters of the comparison NNI series
        Keys:    parameters listed in the input parameter 'parameters'
    radar_plot :  matplotlib figure
        Figure of the generated radar plot

    Raises
    ------
    TypeError
        If an error occurred during the computation of a parameter
    TypeError
        If no input data is provided for the baseline/reference NNI or R-peak series
    TypeError
        If no input data is provided for the comparison NNI or R-peak series
    TypeError
        If no selection of pyHRV parameters is provided
    ValueError
        If less than 2 pyHRV parameters were provided

    Notes
    -----
    ..    If both 'nni' and 'rpeaks' are provided, 'rpeaks' will be chosen over the 'nn' and the 'nni' data will be computed
        from the 'rpeaks'
    ..    If both 'comparison_nni' and 'comparison_rpeaks' are provided, 'comparison_rpeaks' will be chosen over the
        the 'comparison_nni' and the nni data will be computed from the 'comparison_rpeaks'

    """
    # Helper function & variables
    para_func = pyhrv.utils.load_hrv_keys_json()
    unknown_parameters, ref_params, comp_params = [], {}, {}

    def _compute_parameter(nni_series, parameter):

        # Get function name for the requested parameter
        func = para_func[parameter][-1]

        try:
            # Try to pass the show and mode argument to to suppress PSD plots
            index = 0
            if parameter.endswith('_vlf'):
                parameter = parameter.replace('_vlf', '')
            elif parameter.endswith('_lf'):
                index = 1
                parameter = parameter.replace('_lf', '')
            elif parameter.endswith('_hf'):
                index = 2
                parameter = parameter.replace('_hf', '')
            val = eval(func + '(nni=nni_series, mode=\'dev\')[0][\'%s\']' % (parameter))
            #val = val[index]
        except TypeError as e:
            if 'mode' in str(e):
                try:
                    # If functions has now mode feature but 'mode' argument, but a plotting feature
                    val = eval(func + '(nni=nni_series, plot=False)[\'%s\']' % parameter)
                except TypeError as a:
                    if 'plot' in str(a):
                        # If functions has now plotting feature try regular function
                        val = eval(func + '(nni=nni_series)[\'%s\']' % parameter)
                    else:
                        raise TypeError(e)
        return val

    # Check input data
    if nni is None and rpeaks is None:
        raise TypeError(
            "No input data provided for baseline or reference NNI. Please specify the reference NNI series.")
    else:
        nn = pyhrv.utils.check_input(nni, rpeaks)

    if comparison_nni is not None and comparison_rpeaks is not None:
        raise TypeError("No input data provided for comparison NNI. Please specify the comarison NNI series.")
    else:
        comp_nn = pyhrv.utils.check_input(comparison_nni, comparison_rpeaks)

    if parameters is None:
        raise TypeError("No input list of parameters provided for 'parameters'. Please specify a list of the parameters"
                        "to be computed and compared.")
    elif len(parameters) < 2:
        raise ValueError("Not enough parameters selected for a radar chart. Please specify at least 2 HRV parameters "
                         "listed in the 'hrv_keys.json' file.")

    # Check for parameter that require a minimum duration to be computed & remove them if the criteria is not met
    if nn.sum() / 1000. <= 600 or comp_nn.sum() / 1000. <= 600:
        for p in ['sdann', 'sdnn_index']:
            if p in parameters:
                parameters.remove(p)
                warnings.warn("Input NNI series are too short for the computation of the '%s' parameter. This "
                              "parameter has been removed from the parameter list." % p, stacklevel=2)

    # # Register projection of custom RadarAxes class
    register_projection(pyhrv.utils.pyHRVRadarAxes)

    # Check if the provided input parameter exists in pyHRV (hrv_keys.json) & compute available parameters
    for p in parameters:
        p = p.lower()
        if p not in para_func.keys():
            # Save unknown parameters
            unknown_parameters.append(p)
        else:
            # Compute available parameters
            ref_params[p] = _compute_parameter(nn, p)
            comp_params[p] = _compute_parameter(comp_nn, p)

            # Check if any parameters could not be computed (returned as None or Nan) and remove them
            # (avoids visualization artifacts)
            if np.isnan(ref_params[p]) or np.isnan(comp_params[p]):
                ref_params.pop(p)
                comp_params.pop(p)
                warnings.warn("The parameter '%s' could not be computed and has been removed from the parameter list."
                              % p)

    # Raise warning pointing out unknown parameters
    if unknown_parameters != []:
        warnings.warn("Unknown parameters '%s' will not be computed." % unknown_parameters, stacklevel=2)

    # Prepare plot
    colors = ['lightskyblue', 'salmon']
    if legend:
        fig, (ax_l, ax) = plt.subplots(1, 2, figsize=(10, 5), subplot_kw=dict(projection='radar'))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': 'radar'})
    theta = np.linspace(0, 2 * np.pi, len(ref_params.keys()), endpoint=False)
    ax.theta = theta

    # Prepare plot data
    ax.set_varlabels([para_func[s][1].replace(' ', '\n') for s in ref_params.keys()])
    ref_vals = [100 for x in ref_params.keys()]
    com_vals = [
        100 if ref_params[p] == 0 and comp_params.get(p, 0) == 0
        else 100 * 2 if ref_params[p] == 0
        else comp_params.get(p, 0) / ref_params[p] * 100
        for p in ref_params
    ]

    ticks = [0, 50, 75, 100, 150, 200, 400]
    pos = np.arange(len(ticks), dtype=float)  # equal steps

    def forward(y):  # data -> axis coords
        return np.interp(y, ticks, pos)

    def inverse(u):  # axis coords -> data
        return np.interp(u, pos, ticks)

    ax.set_yscale('function', functions=(forward, inverse))
    ax.set_ylim(ticks[0], ticks[-1])  # keep within defined range

    # Now ticks are equally spaced; labels show your real values
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ScalarFormatter())


    # Plot data
    for i, vals in enumerate([ref_vals, com_vals]):
        ax.plot(theta, vals, color=colors[i])
        ax.fill(theta, vals, color=colors[i], alpha=0.3)

    title = 'HRV Parameter Radar Chart'
    # title = "HRV Parameter Radar Chart\nReference NNI Series (%s) vs. Comparison NNI Series (%s)\n" % (
    # colors[0], colors[1]) \
    #         + r"(Chart values in $\%$, Reference NNI parameters $\hat=$100$\%$)"

    # Add legend to second empty plot
    if legend:
        ax_l.set_title(title, horizontalalignment='center')
        legend = []

        # Helper function
        def _add_legend(label, fc="white"):
            return legend.append(mpl.patches.Patch(fc=fc, label="\n" + label))

        # Add list of computed parameters
        _add_legend(reference_label, colors[0])
        for p in ref_params.keys():
            _add_legend("%s:" % para_func[p][1])

        # Add list of comparison parameters
        _add_legend(comparison_label, colors[1])
        for p in ref_params.keys():
            u = para_func[p][2] if para_func[p][2] != "-" else ""
            _add_legend("%.2f%s vs. %.2f%s" % (ref_params[p], u, comp_params[p], u))

        # Add relative differences
        _add_legend("")
        for i, _ in enumerate(ref_params.keys()):
            val = com_vals[i] - 100
            _add_legend("+%.2f%s" % (val, r"$\%$") if val > 0 else "%.2f%s" % (val, r"$\%$"))

        ax_l.legend(handles=legend, ncol=3, frameon=False, loc=7)
        ax_l.axis('off')
    else:
        ax.set_title(title, horizontalalignment='center')

    # Show plot
    if save_path:
        plt.savefig(os.path.join(save_path))
        plt.close('all')
