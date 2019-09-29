
import os

Vm = 10.0
Km = 0.1

for l in [0,100,200,300,400]:

	for d in [10,20,30,40,50]:

		#make directory
		directory = "L" + str(l) + "D" + str(d)
		os.system("mkdir " + directory)

		#make output folder
		os.system("mkdir " + directory + "/output/")

		for Lf in [0.001,0.01,0.1,1.0,10.0]:

			for Kt in [0.001,0.005,0.01,0.05,0.1]:

				#create file
				file = "Lf" + str(Lf) + "Kt" + str(Kt) + ".sh"
				f = open(directory + "/" + file, 'w')

				#write 
				f.write("#!/bin/bash\n")
				f.write("#SBATCH -t 48:00:00\n")
				f.write("#SBATCH --ntasks=1\n")
				f.write("#SBATCH -A ipst-hi\n")
				f.write("#SBATCH --mem-per-cpu=4000\n")
				f.write("\n")

				#Format is: python ../../analysis.py Lf Kt Vm Km d mpR mpL outputDir

				f.write("module load python/2.7.8\n")

				mpR = 2000.0 - l / 2.0
				mpL = l / 2.0 

				execString = "python ~/stochfilaments/analysis.py " + str(Lf) + " " + str(Kt) + " " +\
						     str(Vm) + " " + str(Km) +  " " + str(d) + " " + str(mpR) + " " +\
						     str(mpL) + " ./" + directory + "/output/ &"

				f.write(execString + "\n")
				f.write("wait\n") 






