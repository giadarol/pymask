import shutil

folder_list = [
    'pymask_output_beam1_tuned/sixtrack',
    'pymask_output_beam4_tuned/sixtrack']

for sixtrack_input_folder in folder_list:
    shutil.copy(sixtrack_input_folder + "/fc.2", sixtrack_input_folder + "/fort.2")

    with open(sixtrack_input_folder + "/fort.3", "w") as fout:
        with open("mad/fort_beginning.3", "r") as fid_fort3b:
            fout.write(fid_fort3b.read())
        with open(sixtrack_input_folder + "/fc.3", "r") as fid_fc3:
            fout.write(fid_fc3.read())
        with open("mad/fort_end.3", "r") as fid_fort3e:
            fout.write(fid_fort3e.read())

