#! /bin/bash

mkdir -p collection
for subdir in checkpoints/*
do
    ckpt_name="${subdir##*/}"
    cp $subdir/model.tar.gz collection/$ckpt_name.tar.gz
done