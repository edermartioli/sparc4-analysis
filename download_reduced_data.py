"""
    Created on Aug 14 2024
    
    Description: A tool to download SPARC4 reduced data from lynx server
    
    @author: Eder Martioli <emartioli@lna.br>
    Laboratório Nacional de Astrofísica, Brasil.


    ssh-keygen -t rsa
    ssh-copy-id -i $HOME/.ssh/id_rsa.pub sparc4@200.131.64.47
    
    ssh sparc4@200.131.64.47

    Simple usage examples:


    *** most useful examples ***
    => PHOT and POLAR :
    python download_reduced_data.py --output_dir="/Users/eder/Science/Transits-OPD_2024A" --nights="20240615,20240616,20240617,20240618" --suffixes="stack.fits,lc.fits,ts.fits,polar.fits,db.csv,night_report.txt"
    => PHOT :
    python download_reduced_data.py --output_dir="/Users/eder/Science/Transits-OPD_2024A" --nights="20240604" --suffixes="stack.fits,lc.fits,db.csv,night_report.txt"

    python download_reduced_data.py --output_dir="/Users/eder/Science/Transits-OPD_2024A" --nights="20240605" --objects="HATS-24" --suffixes="stack.fits,lc.fits"
    ****************************


    *** other examples ***
    
    # Example to download all reduced data for a given night
    python download_reduced_data.py --output_dir="/Users/eder/Science/Transits-OPD_2024A" --nights="20240604" --suffixes="stack.fits,lc.fits,db.csv,night_report.txt,ts.fits,polar.fits,proc.fits,.log,MasterZero.fits,MasterDomeFlat.fits"

    # Download reduced data without polarimetry, without proc.fits images, and without calibrations
    python download_reduced_data.py --output_dir="/Users/eder/Science/Transits-OPD_2024A" --nights="20240604" --suffixes="stack.fits,lc.fits,db.csv,night_report.txt"

    # Download reduced data for a given list of objects
    python download_reduced_data.py --output_dir="/Users/eder/Science/Transits-OPD_2024A" --object="WASP-167,HATS-36" --nights="20240604" --suffixes="stack.fits,lc.fits"
    
    
    !!! WARNING !!! -> some suffixes won't be downloaded if using an object list. For example: "proc.fits,db.csv,night_report.txt,.log,MasterZero.fits,MasterDomeFlat.fits"
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import sys, os
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-o", "--output_dir", dest="output_dir", help='Output directory',type='string', default="./")
parser.add_option("-b", "--objects", dest="objects", help='List of object IDs to download data',type='string', default="")
parser.add_option("-c", "--channels", dest="channels", help='Channels to download data',type='string', default="1,2,3,4")
parser.add_option("-i", "--ip", dest="ip", help='Server IP address',type='string', default="200.131.64.47")
parser.add_option("-n", "--nights", dest="nights", help='Night dir',type='string', default="")
parser.add_option("-f", "--suffixes", dest="suffixes", help='List of suffixes to download data',type='string', default="")

parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with download_reduced_data.py -h "); sys.exit(1);

# change lines below to use sshpass if you don't want to set keygen
use_sshpass = False
password = "'secret'"
sshpass = ""
if use_sshpass :
    sshpass = "sshpass -p {} ".format(password)
#########################


nights = options.nights.split(",")
suffixes = options.suffixes.split(",")
channels = options.channels.split(",")

objects = options.objects.split(",")
object_wildcards = []
if len(objects) :
    for obj in objects :
        if obj != "" :
            object_wildcards.append("{}*".format(obj))
        else :
            object_wildcards.append("")

for night in nights :

    outdir = "{}/{}".format(options.output_dir,night)

    # if reduced dir doesn't exist create one
    os.makedirs(outdir, exist_ok=True)

    for suffix in suffixes:
    
        for object_wildcard in object_wildcards :
            if len(channels) == 4 :
                command = "{}scp sparc4@{}:/home/sparc4/reduced/{}/sparc4acs[1-4]/*{}{} {}".format(sshpass,options.ip,night,object_wildcard,suffix,outdir)
            else :
                for channel in channels :
                    command = "{}scp sparc4@{}:/home/sparc4/reduced/{}/sparc4acs{}/*{}{} {}".format(sshpass,options.ip,night,channel,object_wildcard,suffix,outdir)
            
            if use_sshpass :
                print("Running: ", command.replace(password,"******"))
            else :
                print("Running: ", command)

            os.system(command)
