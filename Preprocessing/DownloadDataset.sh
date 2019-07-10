#!/usr/bin/env bash

tmpfile='pr1956/'

aws s3 cp s3://personnel-records/1956/scans/parsed/ $tmpfile --exclude "*" --include pr1956_f4[7-9]_[0-4].tif --recursive
aws s3 cp s3://personnel-records/1956/scans/parsed/ $tmpfile --exclude "*" --include pr1956_f[5-9][0-9]_[0-4].tif --recursive
aws s3 cp s3://personnel-records/1956/scans/parsed/ $tmpfile --exclude "*" --include pr1956_f1[0-3][0-9]_[0-4].tif --recursive
aws s3 cp s3://personnel-records/1956/scans/parsed/ $tmpfile --exclude "*" --include pr1956_f14[0-6]_[0-4].tif --recursive
