#!/bin/bash

foldername=$1

for i in `ls $foldername/`
do count_x=`ls $foldername/$i/*.png -l |wc -l `
count_y=`ls $foldername/$i/y/*.png -l |wc -l `
count=$(($count_x>$count_y?$count_y:$count_x))

echo $count

for (( j=1; j<=$count; j++))
do
mkdir $foldername/$i$j
mkdir $foldername/$i$j/y
mv $foldername/$i/$j.png $foldername/$i$j/1.png
mv $foldername/$i/y/$j.png $foldername/$i$j/y/1.png
done
rm $foldername/$i -R
done

