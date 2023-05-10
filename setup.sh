sudo apt-get update
sudo apt-get install python3-venv
python3 -m venv env # package for virtual environment

source env/bin/activate

pip install -r requirements.txt

#deactivate 