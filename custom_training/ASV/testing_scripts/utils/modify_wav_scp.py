import os
import argparse


if __name__ == "__main__":
    '''
    This code creates the SpeechBrain data manifest starting from a Kaldi formatting from the VPC 2022
    '''

    parser = argparse.ArgumentParser(description='Process input data.')
    parser.add_argument('--file', type=str, help=' Path to the wav.scp file')
    parser.add_argument('--new-loc', type=str, help=' New location for files')
    parser.add_argument('--set-name', type=str, help=' Name of the subset')
    parser.add_argument('--outfolder', type=str, help=' Name of the subset')

    args = parser.parse_args()

    # Read wav.scp
    with open(args.file) as file:
        lines = [line.rstrip() for line in file]
   
    # Modify the file location 
    myfile = open(os.path.join(args.outfolder, args.set_name + ".scp"), 'w')

    for line in lines:
       fields = line.split(" ")
       path = fields[2]
       wavname = os.path.basename(path)
       if args.set_name.split("_")[0] == "vctk" and args.set_name.split("_")[2] == "trials":
           if args.set_name.split("_")[-2] == "f" or args.set_name.split("_")[-3] == "f":
               set = "vctk_{}_trials_f_all".format(args.set_name.split("_")[1]) 
           if args.set_name.split("_")[-2] == "m" or args.set_name.split("_")[-3] == "m":
               set = "vctk_{}_trials_m_all".format(args.set_name.split("_")[1]) 
       else: set = '_'.join(args.set_name.split("_")[:-1])

       if args.set_name.split("_")[2] == "trials" and args.set_name.split("_")[0] == "vctk":
          fields[2] = os.path.join(args.new_loc, "_".join(args.set_name.split("_")[:4])+"_all", wavname)
       else:
          fields[2] = os.path.join(args.new_loc, "_".join(args.set_name.split("_")[:-1]), wavname)
    
       # Save wav.scp
       myfile.write("{}\n".format(' '.join(fields)))
 
