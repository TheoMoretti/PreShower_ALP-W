filename=$1
filename=$2
while read line;do

  for k in {1,2,3,4,5,6};
  do
      echo $line;
      i=( $line )
      mass=${i[0]}
      echo $mass
      echo $k
      python Test_output_multiprocess.py -m $line -c $k
  done

done < $filename$1
