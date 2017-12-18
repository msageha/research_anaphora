#!/bin/sh

for file in *; do
   echo "${file}";
   diff "${file}"  "../../extract_text/OC/${file}";
done