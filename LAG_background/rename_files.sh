#!/bin/bash

foldername=$1
cd $1

find -depth -name '* *' | rename 's/ /_/g'
for i in `ls`;do count=1;for j in `ls $i/*.png`;do mv $j $i/$count.png;count=$((count+1));done;done
for i in `ls`;do count=1;for j in `ls $i/y/*.png`;do mv $j $i/y/$count.png;count=$((count+1));done;done

