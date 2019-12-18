import shutil
import pickle
import numpy as np

import pysixtrack
import sixtracktools

input_to_test = 'pysixtrack_from_pymask'
input_to_test = 'sixtrack_from_pymask'

if input_to_test == 'pysixtrack_from_pymask':
    # Load pysixtrack machine 
    with open("pymask_output_beam1_tuned/pysixtrack/line_bb_dipole_not_cancelled.pkl", "rb") as fid:
        ltest = pysixtrack.Line.from_dict(pickle.load(fid))
elif input_to_test == 'sixtrack_from_pymask':
    sixtrack_input_folder = 'pymask_output_beam1_tuned/sixtrack'
    shutil.copy(sixtrack_input_folder + "/fc.2", sixtrack_input_folder + "/fort.2")

    with open(sixtrack_input_folder + "/fort.3", "w") as fout:
        with open("mad/fort_beginning.3", "r") as fid_fort3b:
            fout.write(fid_fort3b.read())
        with open(sixtrack_input_folder + "/fc.3", "r") as fid_fc3:
            fout.write(fid_fc3.read())
        with open("mad/fort_end.3", "r") as fid_fort3e:
            fout.write(fid_fort3e.read())
    sixinput_test = sixtracktools.sixinput.SixInput(sixtrack_input_folder)
    ltest = pysixtrack.Line.from_sixinput(sixinput_test)
else:
    raise ValueError('What?!')

# Load reference sixtrack machine
sixinput = sixtracktools.sixinput.SixInput("mad/sixtrack/")
lsix = pysixtrack.Line.from_sixinput(sixinput)

original_length = ltest.get_length()
assert (lsix.get_length() - original_length) < 1e-6

# Simplify the two machines
for ll in (ltest, lsix):
    ll.remove_inactive_multipoles(inplace=True)
    ll.remove_zero_length_drifts(inplace=True)
    ll.merge_consecutive_drifts(inplace=True)

# Check that the two machines are identical
assert len(ltest) == len(lsix)

assert (ltest.get_length() - original_length) < 1e-6
assert (lsix.get_length() - original_length) < 1e-6


def norm(x):
    return np.sqrt(np.sum(np.array(x) ** 2))


for ii, (ee_test, ee_six, nn_test, nn_six) in enumerate(
    zip(ltest.elements, lsix.elements, ltest.element_names, lsix.element_names)
):
    assert type(ee_test) == type(ee_six)

    dtest = ee_test.to_dict(keepextra=True)
    dsix = ee_six.to_dict(keepextra=True)

    for kk in dtest.keys():

        # Check if they are identical
        if dtest[kk] == dsix[kk]:
            continue

        # Check if the relative error is small
        try:
            diff_rel = norm(np.array(dtest[kk]) - np.array(dsix[kk])) / norm(
                dtest[kk]
            )
        except ZeroDivisionError:
            diff_rel = 100.0
        if diff_rel < 3e-5:
            continue

        # Check if absolute error is small
        diff_abs = norm(np.array(dtest[kk]) - np.array(dsix[kk]))
        if diff_abs > 0:
            print(f"{nn_test}[{kk}] - test:{dtest[kk]} six:{dsix[kk]}")
        if diff_abs < 1e-12:
            continue

        # Exception: drift length (100 um tolerance)
        if isinstance(
            ee_test, (pysixtrack.elements.Drift, pysixtrack.elements.DriftExact)
        ):
            if kk == "length":
                if diff_abs < 1e-4:
                    continue

        # Exception: multipole lrad is not passed to sixtraxk
        if isinstance(ee_test, pysixtrack.elements.Multipole):
            if kk == "length":
                if np.abs(ee_test.hxl) + np.abs(ee_test.hyl) == 0.0:
                    continue

        # Exceptions BB4D (separations are recalculated)
        if isinstance(ee_test, pysixtrack.elements.BeamBeam4D):
            if kk == "x_bb":
                if diff_abs / dtest["sigma_x"] < 0.0001:
                    continue
            if kk == "y_bb":
                if diff_abs / dtest["sigma_y"] < 0.0001:
                    continue

        # Exceptions BB4D (angles and separations are recalculated)
        if isinstance(ee_test, pysixtrack.elements.BeamBeam6D):
            if kk == "alpha":
                if diff_abs < 10e-6:
                    continue
            if kk == "x_bb_co":
                if diff_abs / np.sqrt(dtest["sigma_11"]) < 0.001:
                    continue
            if kk == "y_bb_co":
                if diff_abs / np.sqrt(dtest["sigma_33"]) < 0.001:
                    continue

        # If it got here it means that no condition above is met
        raise ValueError("Too large discrepancy!")
print(
    """

*******************************************************************
 The line from test seq. and the line from reference are identical!
*******************************************************************
"""
)
