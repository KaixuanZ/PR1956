#!/usr/bin/env bash

sudo mount -t nfs -o nolock,hard 54.183.205.102:/teikoku ../teikoku
sudo mount -t nfs -o nolock,hard 54.183.205.102:/personnel-records ../personnel-records

#umount
##sudo umount ../teikoku
##sudo umount ../personnel-records