import cv2
import os, sys
import numpy as np
import time
from matplotlib import pyplot as plt
import argparse

#from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer