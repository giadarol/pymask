# %%
import gc
import pickle

import numpy as np
from scipy.constants import c as clight

import bb_setup as bbs
import os

bb_data_frames_fname = 'bb_dataframes.pkl'

generate_pysixtrack_lines = True
generate_sixtrack_inputs = True
generate_mad_sequences_with_bb = True

closed_orbit_method_for_pysixtrack = 'from_mad' #'from_mad' or 'from_tracking'

reference_bunch_charge_sixtrack_ppb = 1.2e11
emitnx_sixtrack_um = 2.
emitny_sixtrack_um = 2.
sigz_sixtrack_m = 0.075
sige_sixtrack = 0.00011
flag_ibeco_sixtrack = 1
flag_ibtyp_sixtrack = 0
flag_lhc_sixtrack = 2
flag_ibbc_sixtrack = 0
radius_sixtrack_multip_conversion_mad = 0.017

sequences_to_be_tracked = [
        {'name': 'beam1_tuned', 'fname' : 'mad/lhc_without_bb_fortracking.seq', 'beam': 'b1', 'seqname':'lhcb1', 'bb_df':'bb_df_b1'},
        {'name': 'beam4_tuned', 'fname' : 'mad/lhcb4_without_bb_fortracking.seq', 'beam': 'b2', 'seqname':'lhcb2', 'bb_df':'bb_df_b4'},
       ]

with open(bb_data_frames_fname, 'rb') as fid:
    bb_df_dict = pickle.load(fid)

for ss in sequences_to_be_tracked:

    # Define output folder
    outp_folder = 'pymask_output_' + ss['name']
    os.makedirs(outp_folder, exist_ok=True)

    # Load dataframe and save a copy for reference
    bb_df =  bb_df_dict[ss['bb_df']]
    bb_df.to_pickle(outp_folder+'/bb_df.pkl')

    # Build mad model
    mad_track = bbs.build_mad_instance_with_bb(
        sequences_file_name=ss['fname'],
        bb_data_frames=[bb_df],
        beam_names=[ss['beam']],
        sequence_names=[ss['seqname']],
        mad_echo=False, mad_warn=False, mad_info=False)

    # Explicitly enable bb in mad model
    mad_track.globals.on_bb_switch = 1

    # Get optics and orbit at start ring
    optics_orbit_start_ring = bbs.get_optics_and_orbit_at_start_ring(mad_track, ss['seqname'])
    with open(outp_folder + '/optics_orbit_at_start_ring.pkl', 'wb') as fid:
        pickle.dump(optics_orbit_start_ring, fid)

    # Save mad sequence
    if generate_mad_sequences_with_bb:
        mad_fol_name = outp_folder + '/mad'
        os.makedirs(mad_fol_name, exist_ok=True)
        mad_track.use(ss['seqname'])
        mad_track.input(
                f"save, sequence={ss['seqname']}, beam=true, file={mad_fol_name}/sequence_w_bb.seq")

    # Generate pysixtrack lines
    if generate_pysixtrack_lines:
        pysix_fol_name = outp_folder + "/pysixtrack"
        dct_pysxt = bbs.generate_pysixtrack_line_with_bb(mad_track, ss['seqname'], bb_df,
                closed_orbit_method=closed_orbit_method_for_pysixtrack,
                pickle_lines_in_folder=pysix_fol_name)

    # Generate sixtrack input
    if generate_sixtrack_inputs:
        six_fol_name = outp_folder + "/sixtrack"
        bbs.generate_sixtrack_input(mad_track, ss['seqname'], bb_df, six_fol_name,
        reference_bunch_charge_sixtrack_ppb,
            emitnx_sixtrack_um,
            emitny_sixtrack_um,
            sigz_sixtrack_m,
            sige_sixtrack,
            flag_ibeco_sixtrack,
            flag_ibtyp_sixtrack,
            flag_lhc_sixtrack,
            flag_ibbc_sixtrack,
            radius_sixtrack_multip_conversion_mad)

    # del(mad_track)
    # gc.collect()

# %%
