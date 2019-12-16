# %%
import gc
import pickle

import numpy as np
from scipy.constants import c as clight

import bb_setup as bbs

generate_pysixtrack_lines = True
generate_sixtrack_inputs = False

closed_orbit_to_pysixtrack_from_mad = True

sequence_b1b2_for_optics_fname = 'mad/lhc_without_bb.seq'

sequences_to_be_tracked = [
        {'name': 'beam1_tuned', 'fname' : 'mad/lhc_without_bb_fortracking.seq', 'beam': 'b1', 'seqname':'lhcb1'},
        #{'name': 'beam4_tuned', 'fname' : 'mad/lhcb4_without_bb_fortracking.seq', 'beam': 'b2', 'seqname':'lhcb2'},
       ]
#ip_names = ['ip1', 'ip2', 'ip5', 'ip8']
#numberOfLRPerIRSide = [21, 20, 21, 20]
ip_names = ['ip1', 'ip5']
numberOfLRPerIRSide = [0,0]
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

# Get bb dataframe and mad model (with dummy bb) for beam 4
bb_df_b4 = bbs.get_counter_rotating(bb_df_b2)
bbs.generate_mad_bb_info(bb_df_b4, mode='dummy')

# Generate mad info
bbs.generate_mad_bb_info(bb_df_b1, mode='from_dataframe', madx_reference_bunch_charge=madx_reference_bunch_charge)
bbs.generate_mad_bb_info(bb_df_b2, mode='from_dataframe', madx_reference_bunch_charge=madx_reference_bunch_charge)
bbs.generate_mad_bb_info(bb_df_b4, mode='from_dataframe', madx_reference_bunch_charge=madx_reference_bunch_charge)

# Mad model of the machines to be tracked
for ss in sequences_to_be_tracked:

    bb_df = {'b1': bb_df_b1, 'b2':bb_df_b4}[ss['beam']]

    mad_track = bbs.build_mad_instance_with_bb(
        sequences_file_name=ss['fname'],
        bb_data_frames=[bb_df],
        beam_names=[ss['beam']],
        sequence_names=[ss['seqname']],
        mad_echo=False, mad_warn=False, mad_info=False)

    # disable bb in mad model
    mad_track.globals.on_bb_switch = 0

    # Twiss and get closed-orbit
    mad_track.use(sequence=ss['seqname'])
    twiss_table = mad_track.twiss()

    beta0 = mad_track.sequence[ss['seqname']].beam.beta
    gamma0 = mad_track.sequence[ss['seqname']].beam.gamma
    p0c_eV = mad_track.sequence[ss['seqname']].beam.pc*1.e9

    x_CO  = twiss_table.x[0]
    px_CO = twiss_table.px[0]
    y_CO  = twiss_table.y[0]
    py_CO = twiss_table.py[0]
    t_CO  = twiss_table.t[0]
    pt_CO = twiss_table.pt[0]
    #convert tau, pt to sigma,delta
    sigma_CO = beta0 * t_CO
    delta_CO = ((pt_CO**2 + 2*pt_CO/beta0) + 1)**0.5 - 1

    mad_CO = np.array([x_CO, px_CO, y_CO, py_CO, sigma_CO, delta_CO])

    optics_at_start_ring = {
            'betx': twiss_table.betx[0],
            'bety': twiss_table.betx[0]}

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

        line_for_tracking.disable_beambeam()
        part_on_CO = line_for_tracking.find_closed_orbit(
            guess=mad_CO, p0c=p0c_eV,
            method={True: 'get_guess', False: 'Nelder-Mead'}[closed_orbit_to_pysixtrack_from_mad])
        line_for_tracking.enable_beambeam()

        with open(f"line_{ss['name']}_from_mad.pkl", "wb") as fid:
            linedct = line_for_tracking.to_dict(keepextra=True)
            linedct['particle_on_closed_orbit'] = part_on_CO.to_dict()
            linedct['optics_at_start_ring'] = optics_at_start_ring
            pickle.dump(linedct, fid)

    if generate_sixtrack_inputs:
        raise ValueError('Coming soon :-)')

    # del(mad_track)
    # gc.collect()

# %%
