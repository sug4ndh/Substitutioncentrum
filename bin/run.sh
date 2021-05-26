#!/bin/bash

gunicorn -w 4 -b 127.0.0.1:9008 --log-level=debug unicorn:app
