#!/usr/local/bin/python3.7
import sys

# activate venv
activate_this = '/home/andysee/disasterapp.andysee.com/venv/bin/activate_this.py'

with open(activate_this, "rb") as source_file:
    code = compile(source_file.read(), activate_this, "exec")
exec(code)

import sys
import subprocess
import logging
logging.basicConfig(stream=sys.stderr)

# insert project path to sys path
path = '/home/andysee/disasterapp.andysee.com/app'
if path not in sys.path:
    sys.path.insert(0, path)

from run import app as application
application.secret_key = 'z2VGr1ReRghIdO9dO6HGXbWg3RkWctnd'

