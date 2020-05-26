#!/bin/bash

DATA_ROOT=./

function download () {
  fileurl=${1}
  filename=${fileurl##*/}
  if [ ! -f ${filename} ]; then
    echo ">>> Download '${filename}' from '${fileurl}'."
    wget ${fileurl}
  else
    echo "*** File '${filename}' exists. Skip."
  fi
}

download http://files.grouplens.org/datasets/movielens/ml-100k.zip

unzip ml-100k.zip


