#!/bin/bash
set -eu
ulimit -v 1048576
/usr/bin/time -v -o /tmp/sfora-connected-core-census-run-v1/resources.log timeout 300s /usr/bin/python3 -I -B /tmp/sfora-connected-core-census-run-v1/census_connected_core_errors.py --inputs /tmp/sfora-connected-core-census-inputs-v1/fetch-receipt.json --inputs-sha256 ba98da59fa68676beaea3e20f9f6911e85af3c89163bfe2161c430671ab7af5c --output /tmp/sfora-connected-core-census-run-v1/census.json
