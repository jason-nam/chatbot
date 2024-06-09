from dotenv import load_dotenv
import os
import sys

load_dotenv()

pythonpath = os.getenv('PYTHONPATH')
if pythonpath:
    sys.path.append(pythonpath)

print("PYTHONPATH set to:", pythonpath)
