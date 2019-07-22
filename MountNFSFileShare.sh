#!/usr/bin/env bash

sudo mount -t nfs -o nolock,hard 18.144.66.40:/teikoku ../teikoku
sudo mount -t nfs -o nolock,hard 18.144.66.40:/personnel-records ../personnel-records

#umount
##sudo umount ../teikoku
##sudo umount ../personnel-records