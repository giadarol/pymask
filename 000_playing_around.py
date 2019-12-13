import numpy as np
import pandas as pd
from cpymad.madx import Madx
import smallTempPackage as tp


def BB_FULL(
    circumference=27000,
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
    return myBB

bb_df_b1 = BB_FULL(beam_name='b1', other_beam_name='b2').set_index('elementName', verify_integrity=True).sort_index()
bb_df_b2 = BB_FULL(beam_name='b2', other_beam_name='b1').set_index('elementName', verify_integrity=True).sort_index()

# -------------------------------------------------------------------------------------

def build_mad_instance_with_dummy_bb(sequences_file_name, bb_df,
    beam_names=['b1', 'b2'],
    mad_echo=False, mad_warn=False, mad_info=False):

    mad = Madx()
    mad.options.echo = mad_echo
    mad.options.warn = mad_warn
    mad.options.info = mad_info

    mad.call(sequences_file_name)# assuming a sequence rotated in IR3
    mad.input(bb_df['elementDefinition'].str.cat(sep='\n'))

    # %% seqedit
    for beam in beam_names:
        myBBDFFiltered=bb_df[bb_df['beam']==beam]
        mad.input(f'seqedit, sequence={"lhc"+beam};')
        mad.input('flatten;')
        mad.input(myBBDFFiltered['elementInstallation'].str.cat(sep='\n'))
        mad.input('flatten;')
        mad.input(f'endedit;')

    return mad


sequences_file_name = 'mad/lhc_without_bb.seq'
mad = build_mad_instance_with_dummy_bb(sequences_file_name,
    bb_df=pd.concat([bb_df_b1, bb_df_b2]))

# Check 
data_dict = {}
for beam in ['b1', 'b2']:
    mad.use('lhc'+beam)
    mad.twiss()
    data_dict['twissDF_'+beam]=mad.table.twiss.dframe()

for beam in ['b1', 'b2']:
    twissbeam = data_dict['twissDF_'+beam]
    data_dict['twissDFBB_'+beam]=twissbeam[twissbeam['keyword']=='beambeam']

# Go to Gianni's stuff
from tools import MadPoint, get_bb_names_madpoints_sigmas, compute_shift_strong_beam_based_on_close_ip, find_bb_separations
import tools as bbt

temp_dict={}

for beam, bbdf in zip(['b1', 'b2'], [bb_df_b1, bb_df_b2]):
    # Get locations of the bb encounters (absolute from survey), closed orbit
    # and orientation of the local reference system (MadPoint objects)
    names, positions, sigmas = get_bb_names_madpoints_sigmas(
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

    # DEBUG
    temp_dict[beam] = {'sigmas':sigmas, 'positions':positions, 'names':names}


# Get ip position in the two surveys
ip_names = ['ip1', 'ip2', 'ip5', 'ip8']

ip_position_df = pd.DataFrame()

for beam in ['b1', 'b2']:
    mad.use("lhc"+beam)
    mad.survey()
    for ipnn in ip_names:
        ip_position_df.loc[ipnn, beam] = MadPoint.from_survey((ipnn + ":1").lower(), mad)


# Find partner position and sigmas and correct based on ip
dict_dfs = {'b1': bb_df_b1, 'b2': bb_df_b2}

for self_beam_nn in ['b1', 'b2']:

    self_df = dict_dfs[self_beam_nn]

    for ee in self_df.index:
        other_beam_nn = self_df.loc[ee, 'other_beam']
        other_df = dict_dfs[other_beam_nn]
        other_ee = self_df.loc[ee, 'other_elementName']

        # Get position of the other beam in its own survey
        other_lab_position = other_df.loc[other_ee, 'self_lab_position']

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


# Get ip locations from the survey
ppppp


sep_x, sep_y = find_bb_separations(
    points_weak=bb_xyz_b1,
    points_strong=bb_xyz_b2,
    names=bb_names_b1,
    )
    

