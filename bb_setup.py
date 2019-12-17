import copy

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

