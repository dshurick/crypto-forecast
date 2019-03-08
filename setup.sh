#!/usr/bin/env bash

dvc run -d crypto-forecast/pulldata.py -o data/raw/BTC-USD/ python crypto-forecast/pulldata.py