# %%
import os
import shutil
from cpymad.madx import Madx

mask_fname = "ts_collisions_ats30_en20_IMO380_C7_X160_I1.2_62.3100_60.3200.mask"
sixtrack_input_folder = '../sixtrack'
mylhcbeam = 1
on_bb_switch = 1

mask_fname = "ts_collisions_ats30_en20_IMO380_C7_X160_I1.2_62.3100_60.3200_b4.mask"
sixtrack_input_folder = '../sixtrack_b4'
mylhcbeam = 4
on_bb_switch = 0

os.system("gfortran headonslice.f -o headonslice")

mad = Madx()
mad.globals.mylhcbeam = mylhcbeam
mad.globals.on_bb_switch = on_bb_switch
mad.call(mask_fname)
# mad.input('save, sequence=lhcb1,lhcb2, beam=true, file=lhcwbb.seq;')
# %%
try:
    os.mkdir(sixtrack_input_folder)
except FileExistsError:
    pass

shutil.copy("fc.2", sixtrack_input_folder + "/fort.2")

with open(sixtrack_input_folder + "/fort.3", "w") as fout:
    with open("fort_beginning.3", "r") as fid_fort3b:
        fout.write(fid_fort3b.read())
    with open("fc.3", "r") as fid_fc3:
        fout.write(fid_fc3.read())
    with open("fort_end.3", "r") as fid_fort3e:
        fout.write(fid_fort3e.read())
