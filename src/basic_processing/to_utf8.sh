#!/bin/sh

for file in *; do
   echo "${file}";
   nkf -w --overwrite "${file}";
done