
#!/bin/bash

module load python/2.7.8

for f in {**/*,*}; do

	if [[ $f = Lf*Kt*.sh ]]
	then
   		sbatch $f
	fi

done