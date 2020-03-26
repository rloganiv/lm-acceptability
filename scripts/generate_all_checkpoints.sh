#! /bin/bash

mkdir -p checkpoints/
for config in configs/*
do
    basename="${config##*/}"
    allennlp train $config -s checkpoints/${basename%.*} --include-package acceptability -f
done