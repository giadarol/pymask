import os
import copy
import pickle

import numpy as np
import pandas as pd

from cpymad.madx import Madx

import smallTempPackage as tp
import tools as bbt
from madpoint import MadPoint


def generate_set_of_bb_encounters_1beam(
    circumference=26658.8832,
    harmonic_number = 35640,
    bunch_spacing_buckets = 10,
    numberOfHOSlices = 11,
    bunch_charge_ppb = 0.,
    sigt=0.0755,
    relativistic_beta=1.,
    ip_names = ['ip1', 'ip2', 'ip5', 'ip8'],
    numberOfLRPerIRSide=[21, 20, 21, 20],
    beam_name = 'b1',
    other_beam_name = 'b2'
    ):


    # Long-Range
    myBBLRlist=[]
    for ii, ip_nn in enumerate(ip_names):
        for identifier in (list(range(-numberOfLRPerIRSide[ii],0))+list(range(1,numberOfLRPerIRSide[ii]+1))):
            myBBLRlist.append({'label':'bb_lr', 'ip_name':ip_nn, 'beam':beam_name, 'other_beam':other_beam_name,
                'identifier':identifier})

    if len(myBBLRlist)>0:
        myBBLR=pd.DataFrame(myBBLRlist)[['beam','other_beam','ip_name','label','identifier']]

        myBBLR['self_charge_ppb'] = bunch_charge_ppb
        myBBLR['self_relativistic_beta'] = relativistic_beta
        myBBLR['elementName']=myBBLR.apply(lambda x: tp.elementName(x.label, x.ip_name.replace('ip', ''), x.beam, x.identifier), axis=1)
        myBBLR['other_elementName']=myBBLR.apply(
                lambda x: tp.elementName(x.label, x.ip_name.replace('ip', ''), x.other_beam, x.identifier), axis=1)
        # where circ is used
        BBSpacing = circumference / harmonic_number * bunch_spacing_buckets / 2.
        myBBLR['atPosition']=BBSpacing*myBBLR['identifier']
        # assuming a sequence rotated in IR3
    else:
        myBBLR = pd.DataFrame()

    # Head-On
    numberOfSliceOnSide=int((numberOfHOSlices-1)/2)
    # to check: sigz of the luminous region
    # where sigt is used
    sigzLumi=sigt/2
    z_centroids, z_cuts, N_part_per_slice = tp.constant_charge_slicing_gaussian(1,sigzLumi,numberOfHOSlices)
    myBBHOlist=[]

    for ip_nn in ip_names:
        for identifier in (list(range(-numberOfSliceOnSide,0))+[0]+list(range(1,numberOfSliceOnSide+1))):
            myBBHOlist.append({'label':'bb_ho', 'ip_name':ip_nn, 'other_beam':other_beam_name, 'beam':beam_name, 'identifier':identifier})

    myBBHO=pd.DataFrame(myBBHOlist)[['beam','other_beam', 'ip_name','label','identifier']]


    myBBHO['self_charge_ppb'] = bunch_charge_ppb/numberOfHOSlices
    myBBHO['self_relativistic_beta'] = relativistic_beta
    for ip_nn in ip_names:
        myBBHO.loc[myBBHO['ip_name']==ip_nn, 'atPosition']=list(z_centroids)

    myBBHO['elementName']=myBBHO.apply(lambda x: tp.elementName(x.label, x.ip_name.replace('ip', ''), x.beam, x.identifier), axis=1)
    myBBHO['other_elementName']=myBBHO.apply(lambda x: tp.elementName(x.label, x.ip_name.replace('ip', ''), x.other_beam, x.identifier), axis=1)
    # assuming a sequence rotated in IR3

    myBB=pd.concat([myBBHO, myBBLR],sort=False)
    myBB = myBB.set_index('elementName', drop=False, verify_integrity=True).sort_index()

    return myBB

def generate_mad_bb_info(bb_df, mode='dummy', madx_reference_bunch_charge=1):

    if mode == 'dummy':
        bb_df['elementClass']='beambeam'
        eattributes = lambda charge, label:'sigx = 0.1, '   + \
                    'sigy = 0.1, '   + \
                    'xma  = 1, '     + \
                    'yma  = 1, '     + \
                    f'charge = 0*{charge}, ' +\
                    'slot_id = %d'%({'bb_lr': 4, 'bb_ho': 6}[label]) # need to add 60 for central
        bb_df['elementDefinition']=bb_df.apply(lambda x: tp.elementDefinition(x.elementName, x.elementClass, eattributes(x['self_charge_ppb'], x['label'])), axis=1)
        bb_df['elementInstallation']=bb_df.apply(lambda x: tp.elementInstallation(x.elementName, x.elementClass, x.atPosition, x.ip_name), axis=1)
    elif mode=='from_dataframe':
        bb_df['elementClass']='beambeam'
        eattributes = lambda sigx, sigy, xma, yma, charge, label:f'sigx = {sigx}, '   + \
                    f'sigy = {sigy}, '   + \
                    f'xma  = {xma}, '     + \
                    f'yma  = {yma}, '     + \
                    f'charge := on_bb_switch*{charge}, ' +\
                    'slot_id = %d'%({'bb_lr': 4, 'bb_ho': 6}[label]) # need to add 60 for central
        bb_df['elementDefinition']=bb_df.apply(lambda x: tp.elementDefinition(x.elementName, x.elementClass,
            eattributes(np.sqrt(x['other_Sigma_11']),np.sqrt(x['other_Sigma_33']),
                x['separation_x'], x['separation_y'],
                x['other_charge_ppb']/madx_reference_bunch_charge, x['label'])),
            axis=1)
        bb_df['elementInstallation']=bb_df.apply(lambda x: tp.elementInstallation(x.elementName, x.elementClass, x.atPosition, x.ip_name), axis=1)
    else:
        raise ValueError("mode must be 'dummy' or 'from_dataframe")

    return bb_df

def build_mad_instance_with_bb(sequences_file_name, bb_data_frames,
    beam_names, sequence_names,
    mad_echo=False, mad_warn=False, mad_info=False):

    mad = Madx()
    mad.options.echo = mad_echo
    mad.options.warn = mad_warn
    mad.options.info = mad_info

    mad.call(sequences_file_name)# assuming a sequence rotated in IR3
    for bb_df in bb_data_frames:
        mad.input(bb_df['elementDefinition'].str.cat(sep='\n'))

    # %% seqedit
    for beam, bb_df, seq in zip(beam_names, bb_data_frames, sequence_names):
        myBBDFFiltered=bb_df[bb_df['beam']==beam]
        mad.input(f'seqedit, sequence={"lhc"+beam};')
        mad.input('flatten;')
        mad.input(myBBDFFiltered['elementInstallation'].str.cat(sep='\n'))
        mad.input('flatten;')
        mad.input(f'endedit;')

    return mad

def get_geometry_and_optics_b1_b2(mad, bb_df_b1, bb_df_b2):

    for beam, bbdf in zip(['b1', 'b2'], [bb_df_b1, bb_df_b2]):
        # Get positions of the bb encounters (absolute from survey), closed orbit
        # and orientation of the local reference system (MadPoint objects)
        names, positions, sigmas = bbt.get_bb_names_madpoints_sigmas(
            mad, seq_name="lhc"+beam
        )

        temp_df = pd.DataFrame()
        temp_df['self_lab_position'] = positions
        temp_df['elementName'] = names
        for ss in sigmas.keys():
            temp_df[f'self_Sigma_{ss}'] = sigmas[ss]

        temp_df = temp_df.set_index('elementName', verify_integrity=True).sort_index()

        for cc in temp_df.columns:
            bbdf[cc] = temp_df[cc]

def get_survey_ip_position_b1_b2(mad,
        ip_names = ['ip1', 'ip2', 'ip5', 'ip8']):

    # Get ip position in the two surveys

    ip_position_df = pd.DataFrame()

    for beam in ['b1', 'b2']:
        mad.use("lhc"+beam)
        mad.survey()
        for ipnn in ip_names:
            ip_position_df.loc[ipnn, beam] = MadPoint.from_survey((ipnn + ":1").lower(), mad)

    return ip_position_df

def get_partner_corrected_position_and_optics(bb_df_b1, bb_df_b2, ip_position_df):

    dict_dfs = {'b1': bb_df_b1, 'b2': bb_df_b2}

    for self_beam_nn in ['b1', 'b2']:

        self_df = dict_dfs[self_beam_nn]

        for ee in self_df.index:
            other_beam_nn = self_df.loc[ee, 'other_beam']
            other_df = dict_dfs[other_beam_nn]
            other_ee = self_df.loc[ee, 'other_elementName']

            # Get position of the other beam in its own survey
            other_lab_position = copy.deepcopy(other_df.loc[other_ee, 'self_lab_position'])

            # Compute survey shift based on closest ip
            closest_ip = self_df.loc[ee, 'ip_name']
            survey_shift = (
                    ip_position_df.loc[closest_ip, other_beam_nn].p
                  - ip_position_df.loc[closest_ip, self_beam_nn].p)

            # Shift to reference system of self
            other_lab_position.shift_survey(survey_shift)

            # Store positions
            self_df.loc[ee, 'other_lab_position'] = other_lab_position

            # Get sigmas of the other beam in its own survey
            for ss in bbt._sigma_names:
                self_df.loc[ee, f'other_Sigma_{ss}'] = other_df.loc[other_ee, f'self_Sigma_{ss}']
            # Get charge of other beam
            self_df.loc[ee, 'other_charge_ppb'] = other_df.loc[other_ee, 'self_charge_ppb']
            self_df.loc[ee, 'other_relativistic_beta'] = other_df.loc[other_ee, 'self_relativistic_beta']

def compute_separations(bb_df):

    sep_x, sep_y = bbt.find_bb_separations(
        points_weak=bb_df['self_lab_position'].values,
        points_strong=bb_df['other_lab_position'].values,
        names=bb_df.index.values,
        )

    bb_df['separation_x'] = sep_x
    bb_df['separation_y'] = sep_y

def compute_dpx_dpy(bb_df):
    # Defined as (weak) - (strong)
    for ee in bb_df.index:
        dpx = (bb_df.loc[ee, 'self_lab_position'].tpx
                - bb_df.loc[ee, 'other_lab_position'].tpx)
        dpy = (bb_df.loc[ee, 'self_lab_position'].tpy
                - bb_df.loc[ee, 'other_lab_position'].tpy)

        bb_df.loc[ee, 'dpx'] = dpx
        bb_df.loc[ee, 'dpy'] = dpy

def compute_local_crossing_angle_and_plane(bb_df):

    for ee in bb_df.index:
        alpha, phi = bbt.find_alpha_and_phi(
                bb_df.loc[ee, 'dpx'], bb_df.loc[ee, 'dpy'])

        bb_df.loc[ee, 'alpha'] = alpha
        bb_df.loc[ee, 'phi'] = phi


def get_counter_rotating(bb_df):

    c_bb_df = pd.DataFrame(index=bb_df.index)

    c_bb_df['beam'] = bb_df['beam']
    c_bb_df['other_beam'] = bb_df['other_beam']
    c_bb_df['ip_name'] = bb_df['ip_name']
    c_bb_df['label'] = bb_df['label']
    c_bb_df['identifier'] = bb_df['identifier']
    c_bb_df['elementClass'] = bb_df['elementClass']
    c_bb_df['elementName'] = bb_df['elementName']
    c_bb_df['self_charge_ppb'] = bb_df['self_charge_ppb']
    c_bb_df['other_charge_ppb'] = bb_df['other_charge_ppb']
    c_bb_df['other_elementName'] = bb_df['other_elementName']

    c_bb_df['atPosition'] = bb_df['atPosition'] * (-1.)

    c_bb_df['elementDefinition'] = np.nan
    c_bb_df['elementInstallation'] = np.nan

    c_bb_df['self_lab_position'] = np.nan
    c_bb_df['other_lab_position'] = np.nan

    c_bb_df['self_Sigma_11'] = bb_df['self_Sigma_11'] * (-1.) * (-1.)                  # x * x
    c_bb_df['self_Sigma_12'] = bb_df['self_Sigma_12'] * (-1.) * (-1.) * (-1.)          # x * dx / ds
    c_bb_df['self_Sigma_13'] = bb_df['self_Sigma_13'] * (-1.)                          # x * y
    c_bb_df['self_Sigma_14'] = bb_df['self_Sigma_14'] * (-1.) * (-1.)                  # x * dy / ds
    c_bb_df['self_Sigma_22'] = bb_df['self_Sigma_22'] * (-1.) * (-1.) * (-1.) * (-1.)  # dx / ds * dx / ds
    c_bb_df['self_Sigma_23'] = bb_df['self_Sigma_23'] * (-1.) * (-1.)                  # dx / ds * y
    c_bb_df['self_Sigma_24'] = bb_df['self_Sigma_24'] * (-1.) * (-1.) * (-1.)          # dx / ds * dy / ds
    c_bb_df['self_Sigma_33'] = bb_df['self_Sigma_33']                                  # y * y
    c_bb_df['self_Sigma_34'] = bb_df['self_Sigma_34'] * (-1.)                          # y * dy / ds
    c_bb_df['self_Sigma_44'] = bb_df['self_Sigma_44'] * (-1.) * (-1.)                  # dy / ds * dy / ds

    c_bb_df['other_Sigma_11'] = bb_df['other_Sigma_11'] * (-1.) * (-1.)
    c_bb_df['other_Sigma_12'] = bb_df['other_Sigma_12'] * (-1.) * (-1.) * (-1.)
    c_bb_df['other_Sigma_13'] = bb_df['other_Sigma_13'] * (-1.)
    c_bb_df['other_Sigma_14'] = bb_df['other_Sigma_14'] * (-1.) * (-1.)
    c_bb_df['other_Sigma_22'] = bb_df['other_Sigma_22'] * (-1.) * (-1.) * (-1.) * (-1.)
    c_bb_df['other_Sigma_23'] = bb_df['other_Sigma_23'] * (-1.) * (-1.)
    c_bb_df['other_Sigma_24'] = bb_df['other_Sigma_24'] * (-1.) * (-1.) * (-1.)
    c_bb_df['other_Sigma_33'] = bb_df['other_Sigma_33']
    c_bb_df['other_Sigma_34'] = bb_df['other_Sigma_34'] * (-1.)
    c_bb_df['other_Sigma_44'] = bb_df['other_Sigma_44'] * (-1.) * (-1.)

    c_bb_df['other_relativistic_beta']=bb_df['other_relativistic_beta']
    c_bb_df['separation_x'] = bb_df['separation_x'] * (-1.)
    c_bb_df['separation_y'] = bb_df['separation_y']

    c_bb_df['dpx'] = bb_df['dpx'] * (-1.) * (-1.)
    c_bb_df['dpy'] = bb_df['dpy'] * (-1.)

    # Compute phi and alpha from dpx and dpy
    compute_local_crossing_angle_and_plane(c_bb_df)

    return c_bb_df


def setup_beam_beam_in_line(
    line,
    bb_df,
    bb_coupling=False,
):
    import pysixtrack
    assert bb_coupling is False  # Not implemented

    for ee, eename in zip(line.elements, line.element_names):
        if isinstance(ee, pysixtrack.elements.BeamBeam4D):
            ee.charge = bb_df.loc[eename, 'other_charge_ppb']
            ee.sigma_x = np.sqrt(bb_df.loc[eename, 'other_Sigma_11'])
            ee.sigma_y = np.sqrt(bb_df.loc[eename, 'other_Sigma_33'])
            ee.beta_r = bb_df.loc[eename, 'other_relativistic_beta']
            ee.x_bb = bb_df.loc[eename, 'separation_x']
            ee.y_bb = bb_df.loc[eename, 'separation_y']

        if isinstance(ee, pysixtrack.elements.BeamBeam6D):

            ee.phi = bb_df.loc[eename, 'phi']
            ee.alpha = bb_df.loc[eename, 'alpha']
            ee.x_bb_co = bb_df.loc[eename, 'separation_x']
            ee.y_bb_co = bb_df.loc[eename, 'separation_y']

            ee.charge_slices = [bb_df.loc[eename, 'other_charge_ppb']]
            ee.zeta_slices = [0.0]
            ee.sigma_11 = bb_df.loc[eename, 'other_Sigma_11']
            ee.sigma_12 = bb_df.loc[eename, 'other_Sigma_12']
            ee.sigma_13 = bb_df.loc[eename, 'other_Sigma_13']
            ee.sigma_14 = bb_df.loc[eename, 'other_Sigma_14']
            ee.sigma_22 = bb_df.loc[eename, 'other_Sigma_22']
            ee.sigma_23 = bb_df.loc[eename, 'other_Sigma_23']
            ee.sigma_24 = bb_df.loc[eename, 'other_Sigma_24']
            ee.sigma_33 = bb_df.loc[eename, 'other_Sigma_33']
            ee.sigma_34 = bb_df.loc[eename, 'other_Sigma_34']
            ee.sigma_44 = bb_df.loc[eename, 'other_Sigma_44']

            if not (bb_coupling):
                ee.sigma_13 = 0.0
                ee.sigma_14 = 0.0
                ee.sigma_23 = 0.0
                ee.sigma_24 = 0.0

def get_optics_and_orbit_at_start_ring(mad, seq_name, with_bb_forces=False):

    initial_bb_state = mad.globals.on_bb_switch

    mad.globals.on_bb_switch = {True: 1, False: 0}[with_bb_forces]

    # Twiss and get closed-orbit
    mad.use(sequence=seq_name)
    twiss_table = mad.twiss()

    mad.globals.on_bb_switch = initial_bb_state

    beta0 = mad.sequence[seq_name].beam.beta
    gamma0 = mad.sequence[seq_name].beam.gamma
    p0c_eV = mad.sequence[seq_name].beam.pc*1.e9

    optics_at_start_ring = {
            'beta': beta0,
            'gamma' : gamma0,
            'p0c_eV': p0c_eV,
            'betx': twiss_table.betx[0],
            'bety': twiss_table.bety[0],
            'alfx': twiss_table.alfx[0],
            'alfy': twiss_table.alfy[0],
            'dx': twiss_table.dx[0],
            'dy': twiss_table.dy[0],
            'dpx': twiss_table.dpx[0],
            'dpy': twiss_table.dpy[0],
            'x' : twiss_table.x[0],
            'px' : twiss_table.px[0],
            'y' : twiss_table.y[0],
            'py' : twiss_table.py[0],
            't' : twiss_table.t[0],
            'pt' : twiss_table.pt[0],
            #convert tau, pt to sigma,delta
            'sigma' : beta0 * twiss_table.t[0],
            'delta' : ((twiss_table.pt[0]**2 + 2.*twiss_table.pt[0]/beta0) + 1.)**0.5 - 1.
            }
    return optics_at_start_ring

def generate_pysixtrack_line_with_bb(mad, seq_name, bb_df,
        closed_orbit_method='from_mad', pickle_lines_in_folder=None):

    opt_and_CO = get_optics_and_orbit_at_start_ring(mad, seq_name)

    # Build pysixtrack model
    import pysixtrack
    pysxt_line = pysixtrack.Line.from_madx_sequence(
        mad.sequence[seq_name])

    setup_beam_beam_in_line(pysxt_line, bb_df, bb_coupling=False)

    # Temporary fix due to bug in mad loader
    cavities, cav_names = pysxt_line.get_elements_of_type(
            pysixtrack.elements.Cavity)
    for cc, nn in zip(cavities, cav_names):
        if cc.frequency ==0.:
            ii_mad = mad.sequence[seq_name].element_names().index(nn)
            cc_mad = mad.sequence[seq_name].elements[ii_mad]
            f0_mad = mad.sequence[seq_name].beam.freq0 * 1e6 # mad has it in MHz
            cc.frequency = f0_mad*cc_mad.parent.harmon

    mad_CO = np.array([opt_and_CO[kk] for kk in ['x', 'px', 'y', 'py', 'sigma', 'delta']])

    pysxt_line.disable_beambeam()
    part_on_CO = pysxt_line.find_closed_orbit(
        guess=mad_CO, p0c=opt_and_CO['p0c_eV'],
        method={'from_mad': 'get_guess', 'from_tracking': 'Nelder-Mead'}[closed_orbit_method])
    pysxt_line.enable_beambeam()

    pysxt_line_bb_dipole_cancelled = pysxt_line.copy()

    pysxt_line_bb_dipole_cancelled.beambeam_store_closed_orbit_and_dipolar_kicks(
        part_on_CO,
        separation_given_wrt_closed_orbit_4D=True,
        separation_given_wrt_closed_orbit_6D=True)

    pysxt_dict = {
            'line_bb_dipole_not_cancelled': pysxt_line,
            'line_bb_dipole_cancelled': pysxt_line_bb_dipole_cancelled,
            'particle_on_closed_orbit': part_on_CO}

    if pickle_lines_in_folder is not None:
        pysix_fol_name = pickle_lines_in_folder
        os.makedirs(pysix_fol_name, exist_ok=True)

        with open(pysix_fol_name + "/line_bb_dipole_not_cancelled.pkl", "wb") as fid:
            pickle.dump(pysxt_line.to_dict(keepextra=True), fid)

        with open(pysix_fol_name + "/line_bb_dipole_cancelled.pkl", "wb") as fid:
            pickle.dump(pysxt_line_bb_dipole_cancelled.to_dict(keepextra=True), fid)

        with open(pysix_fol_name + "/particle_on_closed_orbit.pkl", "wb") as fid:
            pickle.dump(part_on_CO.to_dict(), fid)

    return pysxt_dict

def generate_sixtrack_input(mad, seq_name, bb_df, output_folder,
        reference_bunch_charge_sixtrack_ppb,
        emitnx_sixtrack_um,
        emitny_sixtrack_um,
        sigz_sixtrack_m,
        sige_sixtrack,
        ibeco_sixtrack,
        ibtyp_sixtrack,
        lhc_sixtrack,
        ibbc_sixtrack,
        radius_sixtrack_multip_conversion_mad):

    six_fol_name = output_folder
    os.makedirs(six_fol_name, exist_ok=True)

    os.system('rm fc.*')
    mad.use(sequence=seq_name)
    mad.twiss()
    mad.input(f'sixtrack, cavall, radius={radius_sixtrack_multip_conversion_mad}')
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
        + ['NEXT\n'])

    with open(six_fol_name + '/fc.3', 'w') as fid:
        fid.write(f3_string)

