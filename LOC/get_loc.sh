#!/usr/bin/env bash

projects='not_used_full_nams.csv'
i=-1
base='https://github.com/'
temp=temp
out=out3/
ext=.csv

while read project ; do
    echo 'cloning: ' $project
    i=$((i + 1))
    path=$base$project
    echo 'Path is: ' $path
    git clone --depth 1 $path $temp &&
    printf "(Running cloc).. " &&
    cloc $temp --quiet --sum-one --csv > $out$i$ext &&
    printf "(repo will be deleted)\n\n\n"
    rm -rf $temp
done < $projects
