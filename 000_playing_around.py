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
    ip_names = ['IP1', 'IP2', 'IP5', 'IP8']
    ):


    # Long-Range
    myBBLRlist=[]
    for beam in ['b1', 'b2']:
        for ip_nn in ip_names:
            for identifier in (list(range(-numberOfLRPerIRSide,0))+list(range(1,numberOfLRPerIRSide+1))):
                myBBLRlist.append({'label':'bb_lr', 'ip_name':ip_nn, 'beam':beam, 'identifier':identifier})

    myBBLR=pd.DataFrame(myBBLRlist)[['beam','ip_name','label','identifier']]
    myBBLR['elementClass']='beambeam'
    myBBLR['charge [ppb]']=0.
    myBBLR['elementName']=myBBLR.apply(lambda x: tp.elementName(x.label, x.ip_name.replace('IP', ''), x.beam, x.identifier), axis=1)
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
    for beam in ['b1', 'b2']:
        for ip_nn in ip_names:
            for identifier in (list(range(-numberOfSliceOnSide,0))+[0]+list(range(1,numberOfSliceOnSide+1))):
                myBBHOlist.append({'label':'bb_ho', 'ip_name':ip_nn, 'beam':beam, 'identifier':identifier})

    myBBHO=pd.DataFrame(myBBHOlist)[['beam','ip_name','label','identifier']]
    myBBHO['elementClass']='beambeam'
    myBBHO['elementAttributes']=lambda charge:'sigx = 0.1, '   + \
                'sigy = 0.1, '   + \
                'xma  = 1, '     + \
                'yma  = 1, '     + \
                f'charge := {charge}'

    myBBHO['charge [ppb]']=0 
    for ip_nn in ip_names:
        myBBHO.loc[myBBHO['ip_name']==ip_nn, 'atPosition']=list(z_centroids)*2

    myBBHO['elementName']=myBBHO.apply(lambda x: tp.elementName(x.label, x.ip_name.replace('IP', ''), x.beam, x.identifier), axis=1)
    myBBHO['elementDefinition']=myBBHO.apply(lambda x: tp.elementDefinition(x.elementName, x.elementClass, x.elementAttributes(x['charge [ppb]']*0) ), axis=1)
    # assuming a sequence rotated in IR3
    myBBHO['elementInstallation']=myBBHO.apply(lambda x: tp.elementInstallation(x.elementName, x.elementClass, x.atPosition, x.ip_name), axis=1)

    myBB=pd.concat([myBBHO, myBBLR],sort=False)
    return myBB

bb_df = BB_FULL()

# -------------------------------------------------------------------------------------

def build_mad_instance_with_dummy_bb(sequences_file_name, bb_df,
    mad_echo=False, mad_warn=False, mad_info=False):

    mad = Madx()
    mad.options.echo = mad_echo
    mad.options.warn = mad_warn
    mad.options.info = mad_info

    mad.call(sequences_file_name)# assuming a sequence rotated in IR3
    mad.input(bb_df['elementDefinition'].str.cat(sep='\n'))

    # %% seqedit
    for beam in ['b1', 'b2']:
        myBBDFFiltered=bb_df[bb_df['beam']==beam]
        mad.input(f'seqedit, sequence={"lhc"+beam};')
        mad.input('flatten;')
        mad.input(myBBDFFiltered['elementInstallation'].str.cat(sep='\n'))
        mad.input('flatten;')
        mad.input(f'endedit;')
        
    return mad
    

sequences_file_name = 'mad/lhc_without_bb.seq'
mad = build_mad_instance_with_dummy_bb(sequences_file_name, bb_df)

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
from tools import MadPoint, get_bb_names_madpoints_sigmas, shift_strong_beam_based_on_close_ip, find_bb_separations

ip_names = ['IP1', 'IP2', 'IP5', 'IP8'] 
# Get IP locations from the survey
mad.use("lhcb1")
mad.twiss()
mad.survey()
IP_xyz_b1 = {}
for ip in ip_names:
    IP_xyz_b1[ip] = MadPoint.from_survey((ip + ":1").lower(), mad)

mad.use("lhcb2")
mad.twiss()
mad.survey()
IP_xyz_b2 = {}
for ip in ip_names:
    IP_xyz_b2[ip] = MadPoint.from_survey((ip + ":1").lower(), mad)

# Get locations of the bb encounters (absolute from survey), closed orbit
# and orientation of the local reference system (MadPoint objects)
bb_names_b1, bb_xyz_b1, bb_sigmas_b1 = get_bb_names_madpoints_sigmas(
    mad, seq_name="lhcb1"
)
bb_names_b2, bb_xyz_b2, bb_sigmas_b2 = get_bb_names_madpoints_sigmas(
    mad, seq_name="lhcb2"
)

shift_strong_beam_based_on_close_ip(
    points_weak=bb_xyz_b1,
    points_strong=bb_xyz_b2,
    IPs_survey_weak=IP_xyz_b1,
    IPs_survey_strong=IP_xyz_b2,
)

sep_x, sep_y = find_bb_separations(
    points_weak=bb_xyz_b1,
    points_strong=bb_xyz_b2,
    names=bb_names_b1,
    )
    

