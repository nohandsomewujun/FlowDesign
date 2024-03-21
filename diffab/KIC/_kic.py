import pyrosetta
from diffab.KIC.Genkic import GenKic
import os

def _kic(args):
    pdbfilepaths, outputfilepaths, H_chain_names, closure_attempts, min_solutions, filter1, filter2, mode, selector_mode, nums = args
    '''
    selector mode=1 lowest energy
    selector mode=2 random
    filter1: backbond chech
    filter2: pivot rama check
    '''
    pyrosetta.init(' '.join([
        '-mute', 'all',
        '-use_input_sc',
        '-ignore_unrecognized_res',
        '-ignore_zero_occupancy', 'false',
        '-load_PDB_components', 'false',
        '-no_fconfig',
    ]))

    scorefxn = pyrosetta.get_fa_scorefxn()
    for pdbfilepath, outputfilepath, H_chain_name in zip(pdbfilepaths, outputfilepaths, H_chain_names):
        for i in range(nums):
            try:
                input_pose = pyrosetta.pose_from_pdb(os.path.join(pdbfilepath, str(i)+'.pdb'))
            except:
                print(f"{pdbfilepath} can not read pdb file!")
                if os.path.exists(outputfilepath):
                    os.rmdir(outputfilepath)
                break
            # IMGT Scheme
            # 105-117
            H3_start = input_pose.pdb_info().pdb2pose(H_chain_name, 95)
            H3_end = input_pose.pdb_info().pdb2pose(H_chain_name, 102)

            if H3_start == 0 or H3_end == 0:
                print(f"{pdbfilepath} can not locate CDRs")
                if os.path.exists(outputfilepath):
                    os.rmdir(outputfilepath)
                break
            # discard all info between H3_start and H3_end
            gen_kic_input_pose = pyrosetta.Pose()
            gen_kic_input_pose.assign(input_pose)

            loop_residues = [num for num in range(H3_start, H3_end+1)]

            try:
                gk_obj = GenKic(loop_residues=loop_residues)
            except:
                print(f'{pdbfilepath} add loop residue err!')
                if os.path.exists(outputfilepath):
                    os.rmdir(outputfilepath)
                break

            gk_obj.set_closure_attempts(closure_attempts)
            gk_obj.set_min_solutions(min_solutions)

            # random select or energy low select
            gk_obj.set_scorefxn(scorefxn)
            if selector_mode == 1:
                gk_obj.set_selector_type('lowest_energy_selector')
            elif selector_mode == 2:
                gk_obj.set_selector_type('random_selector')

            # set omega to 180
            gk_obj.set_omega_angles()

            for res_num in loop_residues:
                gk_obj.randomize_backbone_by_rama_prepro(res_num)
            gk_obj.close_normal_bond(H3_start, H3_start+1)

            # filter
            if filter1:
                gk_obj.set_filter_loop_bump_check()
            if filter2:
                # take too much time
                for r in gk_obj.pivot_residues:
                    gk_obj.set_filter_rama_prepro(r, cutoff=0.5)

            # close
            try:
                if mode == 1:
                    gk_obj.get_instance().apply(gen_kic_input_pose)
            except:
                print(f'{pdbfilepath} can not apply kic!')
                if os.path.exists(outputfilepath):
                    os.rmdir(outputfilepath)
                break

            gen_kic_input_pose.dump_pdb(os.path.join(outputfilepath, str(i)+'.pdb'))
