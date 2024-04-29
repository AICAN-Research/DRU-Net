#!/bin/bash
isort --sl .
black --line-length 80 .
flake8 .
