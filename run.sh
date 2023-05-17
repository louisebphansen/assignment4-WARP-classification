source env/bin/activate

# Run script with balanced data and different augmentation levels
python3 src/classify.py --epochs 10 --balance balanced --augmentation_level none
python3 src/classify.py --epochs 10 --balance balanced --augmentation_level low
python3 src/classify.py --epochs 10 --balance balanced --augmentation_level high

# Run script with imbalanced data and different augmentation levels
python3 src/classify.py --epochs 10 --balance imbalanced --augmentation_level none
python3 src/classify.py --epochs 10 --balance imbalanced --augmentation_level low
python3 src/classify.py --epochs 10 --balance imbalanced --augmentation_level high