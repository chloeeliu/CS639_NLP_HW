import os
import subprocess

URL = "https://nlp.stanford.edu/data/glove.2024.dolma.300d.zip"
ZIP = "glove.2024.dolma.300d.zip"
TXT = "dolma_300_2024_1.2M.100_combined.txt"

if not os.path.exists(TXT):
    subprocess.run(["curl", "-L", "-o", ZIP, URL])
    subprocess.run(["unzip", ZIP])
