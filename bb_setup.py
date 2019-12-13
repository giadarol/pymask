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
    numberOfLRPerIRSide=21,
    numberOfHOSlices = 11,
    sigt=0.0755,
    ip_names = ['ip1', 'ip2', 'ip5', 'ip8'],
    beam_name = 'b1',
    other_beam_name = 'b2'
    ):


    # Long-Range
    myBBLRlist=[]
    for ip_nn in ip_names:
        for identifier in (list(range(-numberOfLRPerIRSide,0))+list(range(1,numberOfLRPerIRSide+1))):
            myBBLRlist.append({'label':'bb_lr', 'ip_name':ip_nn, 'beam':beam_name, 'other_beam':other_beam_name,
                'identifier':identifier})

    myBBLR=pd.DataFrame(myBBLRlist)[['beam','other_beam','ip_name','label','identifier']]
    myBBLR['elementClass']='beambeam'
    myBBLR['charge [ppb]']=0.
    myBBLR['elementName']=myBBLR.apply(lambda x: tp.elementName(x.label, x.ip_name.replace('ip', ''), x.beam, x.identifier), axis=1)
    myBBLR['other_elementName']=myBBLR.apply(
            lambda x: tp.elementName(x.label, x.ip_name.replace('ip', ''), x.other_beam, x.identifier), axis=1)
    myBBLR['elementClass']='beambeam'
    myBBLR['elementAttributes']=lambda charge:'sigx = 0.1, '   + \
                'sigy = 0.1, '   + \
                'xma  = 1, '     + \
                'yma  = 1, '     + \
                f'charge := {charge}'
    myBBLR['elementDefinition']=myBBLR.apply(lambda x: tp.elementDefinition(x.elementName, x.elementClass, x.elementAttributes(x['charge [ppb]']*0) ), axis=1)
    # where circ is used
    BBSpacing = circumference / harmonic_number * bunch_spacing_buckets / 2.
    myBBLR['atPosition']=BBSpacing*myBBLR['identifier']
    # assuming a sequence rotated in IR3
    myBBLR['elementInstallation']=myBBLR.apply(lambda x: tp.elementInstallation(x.elementName, x.elementClass, x.atPosition, x.ip_name), axis=1)

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
    myBBHO['elementClass']='beambeam'
    myBBHO['elementAttributes']=lambda charge:'sigx = 0.1, '   + \
                'sigy = 0.1, '   + \
                'xma  = 1, '     + \
                'yma  = 1, '     + \
                f'charge := {charge}'

    myBBHO['charge [ppb]']=0
    for ip_nn in ip_names:
        myBBHO.loc[myBBHO['ip_name']==ip_nn, 'atPosition']=list(z_centroids)

    myBBHO['elementName']=myBBHO.apply(lambda x: tp.elementName(x.label, x.ip_name.replace('ip', ''), x.beam, x.identifier), axis=1)
    myBBHO['other_elementName']=myBBHO.apply(lambda x: tp.elementName(x.label, x.ip_name.replace('ip', ''), x.other_beam, x.identifier), axis=1)
    myBBHO['elementDefinition']=myBBHO.apply(lambda x: tp.elementDefinition(x.elementName, x.elementClass, x.elementAttributes(x['charge [ppb]']*0) ), axis=1)
    # assuming a sequence rotated in IR3
    myBBHO['elementInstallation']=myBBHO.apply(lambda x: tp.elementInstallation(x.elementName, x.elementClass, x.atPosition, x.ip_name), axis=1)

    myBB=pd.concat([myBBHO, myBBLR],sort=False)
    myBB = myBB.set_index('elementName', verify_integrity=True).sort_index()

    return myBB

def build_mad_instance_with_dummy_bb(sequences_file_name, bb_data_frames,
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

def compute_separations(bb_df):

    sep_x, sep_y = bbt.find_bb_separations(
        points_weak=bb_df['self_lab_position'].values,
        points_strong=bb_df['other_lab_position'].values,
        names=bb_df.index.values,
        )

    bb_df['separation_x'] = sep_x
    bb_df['separation_y'] = sep_y

def compute_local_crossing_angle_and_plane(bb_df):

    for ee in bb_df.index:
        dpx = bb_df.loc[ee, 'self_lab_position'].tpx - bb_df.loc[ee, 'other_lab_position'].tpx
        dpy = bb_df.loc[ee, 'self_lab_position'].tpy - bb_df.loc[ee, 'other_lab_position'].tpy

        alpha, phi = bbt.find_alpha_and_phi(dpx, dpy)

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
    c_bb_df['elementAttributes'] = np.nan
    c_bb_df['charge [ppb]'] = bb_df['charge [ppb]']
    c_bb_df['other_elementName'] = bb_df['other_elementName']

    c_bb_df['atPosition'] = bb_df['atPosition'] * (-1.)

    c_bb_df['elementDefinition'] = np.nan
    c_bb_df['elementInstallation'] = np.nan

    c_bb_df['self_lab_position'] = np.nan
    c_bb_df['other_lab_position'] = np.nan

    c_bb_df['self_Sigma_11'] = bb_df['self_Sigma_11'] * (-1.) * (-1.)
    c_bb_df['self_Sigma_12'] = bb_df['self_Sigma_12'] * (-1.) * (-1.) * (-1.)
    c_bb_df['self_Sigma_13'] = bb_df['self_Sigma_13'] * (-1.)
    c_bb_df['self_Sigma_14'] = bb_df['self_Sigma_14'] * (-1.) * (-1.)
    c_bb_df['self_Sigma_22'] = bb_df['self_Sigma_22'] * (-1.) * (-1.) * (-1.) * (-1.)
    c_bb_df['self_Sigma_23'] = bb_df['self_Sigma_23'] * (-1.) * (-1.)
    c_bb_df['self_Sigma_24'] = bb_df['self_Sigma_24'] * (-1.) * (-1.) * (-1.)
    c_bb_df['self_Sigma_33'] = bb_df['self_Sigma_33']
    c_bb_df['self_Sigma_34'] = bb_df['self_Sigma_34'] * (-1.)
    c_bb_df['self_Sigma_44'] = bb_df['self_Sigma_44'] * (-1.) * (-1.)

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

    c_bb_df['separation_x'] = bb_df['separation_x'] * (-1.)
    c_bb_df['separation_y'] = bb_df['separation_y']

    c_bb_df['alpha'] = bb_df['alpha'] * (-1.) #???????????
    c_bb_df['phi'] = bb_df['phi'] # ???????

    return c_bb_df
