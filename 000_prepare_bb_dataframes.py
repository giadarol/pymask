# %%
import gc
import pickle

import numpy as np
from scipy.constants import c as clight

import bb_setup as bbs

sequence_b1b2_for_optics_fname = 'mad/lhc_without_bb.seq'

ip_names = ['ip1', 'ip2', 'ip5', 'ip8']
numberOfLRPerIRSide = [21, 20, 21, 20]
circumference = 26658.8832
harmonic_number = 35640
bunch_spacing_buckets = 10
numberOfHOSlices = 11
sigt = 0.075
bunch_charge_ppb = 1.2e11
madx_reference_bunch_charge = 1.2e11
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

# Get bb dataframe and mad model (with dummy bb) for beam 3 and 4
bb_df_b3 = bbs.get_counter_rotating(bb_df_b1)
bb_df_b4 = bbs.get_counter_rotating(bb_df_b2)
bbs.generate_mad_bb_info(bb_df_b3, mode='dummy')
bbs.generate_mad_bb_info(bb_df_b4, mode='dummy')

# Generate mad info
bbs.generate_mad_bb_info(bb_df_b1, mode='from_dataframe', madx_reference_bunch_charge=madx_reference_bunch_charge)
bbs.generate_mad_bb_info(bb_df_b2, mode='from_dataframe', madx_reference_bunch_charge=madx_reference_bunch_charge)
bbs.generate_mad_bb_info(bb_df_b3, mode='from_dataframe', madx_reference_bunch_charge=madx_reference_bunch_charge)
bbs.generate_mad_bb_info(bb_df_b4, mode='from_dataframe', madx_reference_bunch_charge=madx_reference_bunch_charge)

# Save to file
with open('bb_dataframes.pkl', 'wb') as fid:
    pickle.dump({
        'bb_df_b1': bb_df_b1,
        'bb_df_b2': bb_df_b2,
        'bb_df_b3': bb_df_b3,
        'bb_df_b4': bb_df_b4,
        'temp_rf_frequency': harmonic_number*relativistic_beta*clight/circumference,
        }, fid)
