#!/usr/bin/env zsh

for FILE (code/*/*.txt) { 
    echo $FILE
    cat header.cpp $FILE > $FILE.cpp
    sed -i 's/void main/int main/' $FILE.cpp
}
