#!/bin/bash

nohup sh $1 2>&1 | tee output.log &