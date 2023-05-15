source env/bin/activate

python3 src/classify_balanced.py --epochs 10 --balanced True --augmentation_level none
python3 src/classify_balanced.py --epochs 10 --balanced True --augmentation_level low
python3 src/classify_balanced.py --epochs 10 --balanced True --augmentation_level high

python3 src/classify_balanced.py --epochs 10 --balanced True --augmentation_level none
python3 src/classify_balanced.py --epochs 10 --balanced True --augmentation_level low
python3 src/classify_balanced.py --epochs 10 --balanced True --augmentation_level high