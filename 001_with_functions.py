import bb_setup as bbs

sequence_fname = 'mad/lhc_without_bb.seq' 
ip_names = ['ip1', 'ip2', 'ip5', 'ip8']

bb_df_b1 = bbs.generate_set_of_bb_encounters_1beam(
    circumference=26658.8832,
    harmonic_number = 35640,
    bunch_spacing_buckets = 10,
    numberOfLRPerIRSide=21,
    numberOfHOSlices = 11,
    sigt=0.0755,
    ip_names = ip_names,
    beam_name = 'b1',
    other_beam_name = 'b2')


bb_df_b2 = bbs.generate_set_of_bb_encounters_1beam(
    circumference=26658.8832,
    harmonic_number = 35640,
    bunch_spacing_buckets = 10,
    numberOfLRPerIRSide=21,
    numberOfHOSlices = 11,
    sigt=0.0755,
    ip_names = ip_names,
    beam_name = 'b2',
    other_beam_name = 'b1')


mad = bbs.build_mad_instance_with_dummy_bb(
    sequences_file_name=sequence_fname,
    bb_data_frames=[bb_df_b1, bb_df_b2],
    beam_names=['b1', 'b2'],
    sequence_names=['lhcb1', 'lhcb2'],
    mad_echo=False, mad_warn=False, mad_info=False)

bbs.get_geometry_and_optics_b1_b2(mad, bb_df_b1, bb_df_b2)

ip_position_df = bbs.get_survey_ip_position_b1_b2(mad, ip_names)

bbs.get_partner_corrected_position_and_optics(
        bb_df_b1, bb_df_b2, ip_position_df)

for bb_df in [bb_df_b1, bb_df_b2]:
    bbs.compute_separations(bb_df)
    bbs.compute_local_crossing_angle_and_plane(bb_df)
