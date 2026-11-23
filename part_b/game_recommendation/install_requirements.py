"""Install the packages for this project using the Python running this file."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    requirements = Path(__file__).resolve().with_name('requirements.txt')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(requirements)])
    print('\nInstallation complete. Run game_recommendation.py next.')
