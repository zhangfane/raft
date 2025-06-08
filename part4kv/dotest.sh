#!/bin/bash
set -x
set -e

logfile=/Users/zhangfan/GolandProjects/raft/temp/rlog

go test -v -race -run "$@" 2>&1 | tee ${logfile}

go run ../tools/raft-testlog-viz/main.go < ${logfile}
