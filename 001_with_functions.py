# %%
import gc
import pickle

import numpy as np
from scipy.constants import c as clight

import bb_setup as bbs

generate_pysixtrack_lines = True
generate_sixtrack_inputs = False

sequence_b1b2_for_optics_fname = 'mad/lhc_without_bb.seq'

sequences_to_be_tracked = [
        {'name': 'beam1_tuned', 'fname' : 'mad/lhc_without_bb_fortracking.seq', 'beam': 'b1', 'seqname':'lhcb1'},
        {'name': 'beam4_tuned', 'fname' : 'mad/lhcb4_without_bb_fortracking.seq', 'beam': 'b2', 'seqname':'lhcb2'},
       ]
ip_names = ['ip1', 'ip2', 'ip5', 'ip8']
numberOfLRPerIRSide = [21, 20, 21, 20]
circumference = 26658.8832
harmonic_number = 35640
bunch_spacing_buckets = 10
numberOfHOSlices = 11
sigt = 0.075
bunch_charge_ppb = 1.2e11
relativistic_gamma=6927.628061781486
relativistic_beta = np.sqrt(1 - 1.0 / relativistic_gamma ** 2)

# Generate dataframes with names and location of the bb encounters (B1)
bb_df_b1 = bbs.generate_set_of_bb_encounters_1beam(
    circumference, harmonic_number,
    bunch_spacing_buckets,
    numberOfHOSlices, bunch_charge_ppb, sigt,
    relativistic_beta, ip_names, numberOfLRPerIRSide,
    beam_name = 'b1',
    other_beam_name = 'b2')

# Generate dataframes with names and location of the bb encounters (B2)
bb_df_b2 = bbs.generate_set_of_bb_encounters_1beam(
    circumference, harmonic_number,
    bunch_spacing_buckets,
    numberOfHOSlices, bunch_charge_ppb, sigt,
    relativistic_beta, ip_names, numberOfLRPerIRSide,
    beam_name = 'b2',
    other_beam_name = 'b1')

# Generate mad info
bbs.generate_mad_bb_info(bb_df_b1, mode='dummy')
bbs.generate_mad_bb_info(bb_df_b2, mode='dummy')

# Install dummy bb lenses in mad sequences
mad = bbs.build_mad_instance_with_bb(
    sequences_file_name=sequence_b1b2_for_optics_fname,
    bb_data_frames=[bb_df_b1, bb_df_b2],
    beam_names=['b1', 'b2'],
    sequence_names=['lhcb1', 'lhcb2'],
    mad_echo=False, mad_warn=False, mad_info=False)

# Use mad survey and twiss to get geometry and locations of all encounters
bbs.get_geometry_and_optics_b1_b2(mad, bb_df_b1, bb_df_b2)

# Get the position of the IPs in the surveys of the two beams
ip_position_df = bbs.get_survey_ip_position_b1_b2(mad, ip_names)

# # Done with this madx model (we free some memory)
# del(mad)
# gc.collect()

# Get geometry and optics at the partner encounter
bbs.get_partner_corrected_position_and_optics(
        bb_df_b1, bb_df_b2, ip_position_df)

# Compute separation, crossing plane rotation and crossing angle
for bb_df in [bb_df_b1, bb_df_b2]:
    bbs.compute_separations(bb_df)
    bbs.compute_dpx_dpy(bb_df)
    bbs.compute_local_crossing_angle_and_plane(bb_df)

# Get bb dataframe and mad model (with dummy bb) for beam 4
bb_df_b4 = bbs.get_counter_rotating(bb_df_b2)
bbs.generate_mad_bb_info(bb_df_b4, mode='dummy')


# Mad model of the machines to be tracked (bb is still dummy)
for ss in sequences_to_be_tracked:

    bb_df = {'b1': bb_df_b1, 'b2':bb_df_b4}[ss['beam']]

    mad_track = bbs.build_mad_instance_with_bb(
        sequences_file_name=ss['fname'],
        bb_data_frames=[bb_df],
        beam_names=[ss['beam']],
        sequence_names=[ss['seqname']],
        mad_echo=False, mad_warn=False, mad_info=False)

    if generate_pysixtrack_lines:
        # Build pysixtrack model
        import pysixtrack
        line_for_tracking = pysixtrack.Line.from_madx_sequence(
            mad_track.sequence[ss['seqname']])

        bbs.setup_beam_beam_in_line(line_for_tracking, bb_df, bb_coupling=False)

        # Temporary fix due to bug in loader
        cavities, _ = line_for_tracking.get_elements_of_type(
                pysixtrack.elements.Cavity)
        for cc in cavities:
            cc.frequency = harmonic_number*relativistic_beta*clight/circumference

        with open(f"line_{ss['name']}_from_mad.pkl", "wb") as fid:
            pickle.dump(line_for_tracking.to_dict(keepextra=True), fid)

    if generate_sixtrack_inputs:
        raise ValueError('Coming soon :-)')

    del(mad_track)
    gc.collect()

# %%
