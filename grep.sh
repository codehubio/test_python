#!/bin/bash
if grep -q "$1" good_acc_files_list.txt; then
  echo $1
fi