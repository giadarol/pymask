# %%
import os
import shutil
from cpymad.madx import Madx

#os.system("gfortran headonslice.f -o headonslice")

mad = Madx()
mad.globals.mylhcbeam = 1
mad.globals.on_bb_switch = 1
mad.call("ts_collisions_ats30_en20_IMO380_C7_X160_I1.2_62.3100_60.3200.mask")
# mad.input('save, sequence=lhcb1,lhcb2, beam=true, file=lhcwbb.seq;')
# %%
try:
    os.mkdir("../sixtrack")
except FileExistsError:
    pass

shutil.copy("fc.2", "../sixtrack/fort.2")

with open("../sixtrack/fort.3", "w") as fout:
    with open("fort_beginning.3", "r") as fid_fort3b:
        fout.write(fid_fort3b.read())
    with open("fc.3", "r") as fid_fc3:
        fout.write(fid_fc3.read())
    with open("fort_end.3", "r") as fid_fort3e:
        fout.write(fid_fort3e.read())
