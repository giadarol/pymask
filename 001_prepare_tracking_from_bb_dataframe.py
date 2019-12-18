# %%
import gc
import pickle

import numpy as np
from scipy.constants import c as clight

import bb_setup as bbs
import os

bb_data_frames_fname = 'bb_dataframes.pkl'

generate_pysixtrack_lines = False
generate_sixtrack_inputs = True
generate_mad_sequences_with_bb = True

closed_orbit_method_for_pysixtrack = 'from_mad' #'from_mad' or 'from_tracking'

reference_bunch_charge_sixtrack_ppb = 1.2e11
emitnx_sixtrack_um = 2.
emitny_sixtrack_um = 2.
sigz_sixtrack_m = 0.075
sige_sixtrack = 0.00011
ibeco_sixtrack = 1
ibtyp_sixtrack = 0
lhc_sixtrack = 2
ibbc_sixtrack = 0
radius_sixtrack_multip_conversion_mad = 0.017

sequences_to_be_tracked = [
        {'name': 'beam1_tuned', 'fname' : 'mad/lhc_without_bb_fortracking.seq', 'beam': 'b1', 'seqname':'lhcb1', 'bb_df':'bb_df_b1'},
        #{'name': 'beam4_tuned', 'fname' : 'mad/lhcb4_without_bb_fortracking.seq', 'beam': 'b2', 'seqname':'lhcb2', 'bb_df':'bb_df_b4'},
       ]

with open(bb_data_frames_fname, 'rb') as fid:
    bb_df_dict = pickle.load(fid)

# Mad model of the machines to be tracked
for ss in sequences_to_be_tracked:

    bb_df =  bb_df_dict[ss['bb_df']]

    outp_folder = 'pymask_output_' + ss['name']
    mad_fol_name = outp_folder + '/mad'
    os.makedirs(mad_fol_name, exist_ok=True)
    bb_df.to_pickle(outp_folder+'/bb_df.pkl')

    mad_track = bbs.build_mad_instance_with_bb(
        sequences_file_name=ss['fname'],
        bb_data_frames=[bb_df],
        beam_names=[ss['beam']],
        sequence_names=[ss['seqname']],
        mad_echo=False, mad_warn=False, mad_info=False)

    # explicitly enable bb in mad model
    mad_track.globals.on_bb_switch = 1

    # Optics and orbit at start ring
    optics_orbit_start_ring = bbs.get_optics_and_orbit_at_start_ring(mad_track, seq_name)
    with open(outp_folder + '/optics_orbit_at_start_ring.pkl', 'wb'):
        pickle.dump(optics_orbit_start_ring, fid)

    # Save mad sequence
    if generate_mad_sequences_with_bb:
        mad_track.use(ss['seqname'])
        mad_track.input(
                f"save, sequence={ss['seqname']}, beam=true, file={mad_fol_name}/sequence_w_bb.seq")

    # Generate pysixtrack lines
    if generate_pysixtrack_lines:
        pysix_fol_name = outp_folder + "/pysixtrack"
        os.makedirs(pysix_fol_name, exist_ok=True)

        dct_pysxt = bbs.generate_pysixtrack_line_with_bb(mad_track, ss['seqname'], bb_df,
                closed_orbit_method=closed_orbit_method_for_pysixtrack,
                pickle_lines_in_folder=pysix_fol_name)

    # Generate sixtrack input
    if generate_sixtrack_inputs:
        six_fol_name = outp_folder + "/sixtrack"
        os.makedirs(six_fol_name, exist_ok=True)

        os.system('rm fc.*')
        mad_track.use(sequence=ss['seqname'])
        mad_track.twiss()
        mad_track.input(f'sixtrack, cavall, radius={radius_sixtrack_multip_conversion_mad}')
        os.system(f'mv fc.* {six_fol_name}')
        os.system(f'cp {six_fol_name}/fc.2 {six_fol_name}/fc.2.old')

        with open(six_fol_name + '/fc.2', 'r') as fid:
            fc2lines = fid.readlines()

        for ii, ll in enumerate(fc2lines):
            llfields = ll.split()
            try:
                if int(llfields[1]) == 20:
                    newll = ' '.join([
                        llfields[0],
                        llfields[1]]
                        + (len(llfields)-2)* ['0.0']
                        +['\n'])
                    fc2lines[ii] = newll
            except ValueError:
                pass # line does not have an integer in the second field
            except IndexError:
                pass # line has less than two fields

        with open(six_fol_name + '/fc.2', 'w') as fid:
            fid.writelines(fc2lines)

        # http://sixtrack.web.cern.ch/SixTrack/docs/user_full/manual.php#Ch6.S6

        sxt_df_4d = bb_df[bb_df['label']=='bb_lr'].copy()
        sxt_df_4d['h-sep [mm]'] = -sxt_df_4d['separation_x']*1e3
        sxt_df_4d['v-sep [mm]'] = -sxt_df_4d['separation_y']*1e3
        sxt_df_4d['strength-ratio'] = sxt_df_4d['other_charge_ppb']/reference_bunch_charge_sixtrack_ppb
        sxt_df_4d['4dSxx [mm*mm]'] = sxt_df_4d['other_Sigma_11']*1e6
        sxt_df_4d['4dSyy [mm*mm]'] = sxt_df_4d['other_Sigma_33']*1e6
        sxt_df_4d['4dSxy [mm*mm]'] = sxt_df_4d['other_Sigma_13']*1e6
        sxt_df_4d['fort3entry'] = sxt_df_4d.apply(lambda x: ' '.join([
                f"{x.elementName}",
                '0',
                f"{x['4dSxx [mm*mm]']}",
                f"{x['4dSyy [mm*mm]']}",
                f"{x['h-sep [mm]']}",
                f"{x['v-sep [mm]']}",
                f"{x['strength-ratio']}",
                # f"{x['4dSxy [mm*mm]']}" Not really used
                ]), axis=1)


        sxt_df_6d = bb_df[bb_df['label']=='bb_ho'].copy()
        sxt_df_6d['h-sep [mm]'] = -sxt_df_6d['separation_x']*1e3
        sxt_df_6d['v-sep [mm]'] = -sxt_df_6d['separation_y']*1e3
        sxt_df_6d['phi [rad]'] = sxt_df_6d['phi']
        sxt_df_6d['alpha [rad]'] = sxt_df_6d['alpha']
        sxt_df_6d['strength-ratio'] = sxt_df_6d['other_charge_ppb']/reference_bunch_charge_sixtrack_ppb
        sxt_df_6d['Sxx [mm*mm]'] = sxt_df_6d['other_Sigma_11'] *1e6
        sxt_df_6d['Sxxp [mm*mrad]'] = sxt_df_6d['other_Sigma_12'] *1e6
        sxt_df_6d['Sxpxp [mrad*mrad]'] = sxt_df_6d['other_Sigma_22'] *1e6
        sxt_df_6d['Syy [mm*mm]'] = sxt_df_6d['other_Sigma_33'] *1e6
        sxt_df_6d['Syyp [mm*mrad]'] = sxt_df_6d['other_Sigma_34'] *1e6
        sxt_df_6d['Sypyp [mrad*mrad]'] = sxt_df_6d['other_Sigma_44'] *1e6
        sxt_df_6d['Sxy [mm*mm]'] = sxt_df_6d['other_Sigma_13'] *1e6
        sxt_df_6d['Sxyp [mm*mrad]'] = sxt_df_6d['other_Sigma_14'] *1e6
        sxt_df_6d['Sxpy [mrad*mm]'] = sxt_df_6d['other_Sigma_23'] *1e6
        sxt_df_6d['Sxpyp [mrad*mrad]'] = sxt_df_6d['other_Sigma_24'] *1e6
        sxt_df_6d['fort3entry'] = sxt_df_6d.apply(lambda x: ' '.join([
                f"{x.elementName}",
                '1',
                f"{x['phi [rad]']}",
                f"{x['alpha [rad]']}",
                f"{x['h-sep [mm]']}",
                f"{x['v-sep [mm]']}",
                '\n'
                f"{x['Sxx [mm*mm]']}",
                f"{x['Sxxp [mm*mrad]']}",
                f"{x['Sxpxp [mrad*mrad]']}",
                f"{x['Syy [mm*mm]']}",
                f"{x['Syyp [mm*mrad]']}",
                '\n',
                f"{x['Sypyp [mrad*mrad]']}",
                f"{x['Sxy [mm*mm]']}",
                f"{x['Sxyp [mm*mrad]']}",
                f"{x['Sxpy [mrad*mm]']}",
                f"{x['Sxpyp [mrad*mrad]']}",
                f"{x['strength-ratio']}",
                ]), axis=1)

        f3_common_settings = ' '.join([
                f"{reference_bunch_charge_sixtrack_ppb}",
                f"{emitnx_sixtrack_um}",
                f"{emitny_sixtrack_um}",
                f"{sigz_sixtrack_m}",
                f"{sige_sixtrack}",
                f"{ibeco_sixtrack}",
                f"{ibtyp_sixtrack}",
                f"{lhc_sixtrack}",
                f"{ibbc_sixtrack}",
                ])

        f3_string = '\n'.join([
            'BEAM',
            'EXPERT',
            f3_common_settings]
            + list(sxt_df_6d['fort3entry'].values)
            + list(sxt_df_4d['fort3entry'].values)
            + ['NEXT'])

        with open(six_fol_name + '/fc.3', 'w') as fid:
            fid.write(f3_string)


    # del(mad_track)
    # gc.collect()

# %%
