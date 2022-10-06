#!/bin/bash

for file in noise/*.png; do
    echo "include_bytes!(\"$file\"),"
done