import numpy as np
from madpoint import MadPoint

_sigma_names = [11, 12, 13, 14, 22, 23, 24, 33, 34, 44]
_beta_names = ["betx", "bety"]


def norm(v):
    return np.sqrt(np.sum(v ** 2))


def get_points_twissdata_for_elements(
    ele_names, mad, seq_name, use_survey=True, use_twiss=True
):

    mad.use(sequence=seq_name)

    mad.twiss()

    if use_survey:
        mad.survey()

    bb_xyz_points = []
    bb_twissdata = {
        kk: []
        for kk in _sigma_names
        + _beta_names
        + "dispersion_x dispersion_y x y".split()
    }
    for eename in ele_names:
        bb_xyz_points.append(
            MadPoint(
                eename + ":1", mad, use_twiss=use_twiss, use_survey=use_survey
            )
        )

        i_twiss = np.where(mad.table.twiss.name == (eename + ":1"))[0][0]

        for sn in _sigma_names:
            bb_twissdata[sn].append(
                getattr(mad.table.twiss, "sig%d" % sn)[i_twiss]
            )

        for kk in ["betx", "bety"]:
            bb_twissdata[kk].append(mad.table.twiss[kk][i_twiss])
        gamma = mad.table.twiss.summary.gamma
        beta = np.sqrt(1.0 - 1.0 / (gamma * gamma))
        for pp in ["x", "y"]:
            bb_twissdata["dispersion_" + pp].append(
                mad.table.twiss["d" + pp][i_twiss] * beta
            )
            bb_twissdata[pp].append(mad.table.twiss[pp][i_twiss])
        # , 'dx', 'dy']:

    return bb_xyz_points, bb_twissdata


def get_elements(seq, ele_type=None, slot_id=None):

    elements = []
    element_names = []
    for ee in seq.elements:

        if ele_type is not None:
            if ee.base_type.name != ele_type:
                continue

        if slot_id is not None:
            if ee.slot_id != slot_id:
                continue

        elements.append(ee)
        element_names.append(ee.name)

    return elements, element_names


def get_points_twissdata_for_element_type(
    mad, seq_name, ele_type=None, slot_id=None, use_survey=True, use_twiss=True
):

    elements, element_names = get_elements(
        seq=mad.sequence[seq_name], ele_type=ele_type, slot_id=slot_id
    )

    points, twissdata = get_points_twissdata_for_elements(
        element_names,
        mad,
        seq_name,
        use_survey=use_survey,
        use_twiss=use_twiss,
    )

    return elements, element_names, points, twissdata


###############################
# beam beam related functions #
###############################

def find_alpha_and_phi(dpx, dpy):

    absphi = np.sqrt(dpx ** 2 + dpy ** 2) / 2.0

    if absphi < 1e-20:
        phi = absphi
        alpha = 0.0
    else:
        if dpy>=0.:
            if dpx>=0:
                # First quadrant
                if np.abs(dpx) >= np.abs(dpy):
                    # First octant
                    phi = absphi
                    alpha = np.arctan(dpy/dpx)
                else:
                    # Second octant
                    phi = absphi
                    alpha = 0.5*np.pi - np.arctan(dpx/dpy)
            else: #dpx<0
                # Second quadrant
                if np.abs(dpx) <  np.abs(dpy):
                    # Third octant
                    phi = absphi
                    alpha = 0.5*np.pi - np.arctan(dpx/dpy)
                else:
                    # Forth  octant
                    phi = -absphi
                    alpha = np.arctan(dpy/dpx)
        else: #dpy<0
            if dpx<=0:
                # Third quadrant
                if np.abs(dpx) >= np.abs(dpy):
                    # Fifth octant
                    phi = -absphi
                    alpha = np.arctan(dpy/dpx)
                else:
                    # Sixth octant
                    phi = -absphi
                    alpha = 0.5*np.pi - np.arctan(dpx/dpy)
            else: #dpx>0
                # Forth quadrant
                if np.abs(dpx) <= np.abs(dpy):
                    # Seventh octant
                    phi = -absphi
                    alpha = 0.5*np.pi - np.arctan(dpx/dpy)
                else:
                    # Eighth octant
                    phi = absphi
                    alpha = np.arctan(dpy/dpx)

    return alpha, phi



def get_bb_names_madpoints_sigmas(
    mad, seq_name, use_survey=True, use_twiss=True
):
    (
        _,
        element_names,
        points,
        twissdata,
    ) = get_points_twissdata_for_element_type(
        mad,
        seq_name,
        ele_type="beambeam",
        slot_id=None,
        use_survey=use_survey,
        use_twiss=use_twiss,
    )
    sigmas = {kk: twissdata[kk] for kk in _sigma_names}
    return element_names, points, sigmas


def compute_shift_strong_beam_based_on_close_ip(
    points_weak, points_strong, IPs_survey_weak, IPs_survey_strong
):
    strong_shift = []
    for i_bb, _ in enumerate(points_weak):

        pbw = points_weak[i_bb]
        pbs = points_strong[i_bb]

        # Find closest IP
        d_ip = 1e6
        use_ip = 0
        for ip in IPs_survey_weak.keys():
            dd = norm(pbw.p - IPs_survey_weak[ip].p)
            if dd < d_ip:
                use_ip = ip
                d_ip = dd

        # Shift Bs
        shift_ws = IPs_survey_strong[use_ip].p - IPs_survey_weak[use_ip].p
        strong_shift.append(shift_ws)
    return strong_shift


def find_bb_separations(points_weak, points_strong, names=None):

    if names is None:
        names = ["bb_%d" % ii for ii in range(len(points_weak))]

    sep_x = []
    sep_y = []
    for i_bb, name_bb in enumerate(names):

        pbw = points_weak[i_bb]
        pbs = points_strong[i_bb]

        # Find vws
        vbb_ws = points_strong[i_bb].p - points_weak[i_bb].p

        # Check that the two reference system are parallel
        try:
            assert norm(pbw.ex - pbs.ex) < 1e-10  # 1e-4 is a reasonable limit
            assert norm(pbw.ey - pbs.ey) < 1e-10  # 1e-4 is a reasonable limit
            assert norm(pbw.ez - pbs.ez) < 1e-10  # 1e-4 is a reasonable limit
        except AssertionError:
            print(name_bb, "Reference systems are not parallel")
            if (
                np.sqrt(
                    norm(pbw.ex - pbs.ex) ** 2
                    + norm(pbw.ey - pbs.ey) ** 2
                    + norm(pbw.ez - pbs.ez) ** 2
                )
                < 5e-3
            ):
                print("Smaller that 5e-3, tolerated.")
            else:
                raise ValueError("Too large! Stopping.")

        # Check that there is no longitudinal separation
        try:
            assert np.abs(np.dot(vbb_ws, pbw.ez)) < 1e-4
        except AssertionError:
            print(name_bb, "The beams are longitudinally shifted")

        # Find separations
        sep_x.append(np.dot(vbb_ws, pbw.ex))
        sep_y.append(np.dot(vbb_ws, pbw.ey))

    return sep_x, sep_y


