cd Warp-C

mkdir train
cd train

mkdir plastic
mkdir glass 
mkdir cardboard 
mkdir metal

cd .. 

# add to plastic folder
for f in train_crops/bottle/bottle*/*; do
    cp -v "$f" train/plastic/"${f//\//_}"
done

for f in train_crops/canister/*/*; do
    cp -v "$f" train/plastic/"${f//\//_}"
done

for f in train_crops/detergent/detergent*/*; do
    cp -v "$f" train/plastic/"${f//\//_}"
done

# add to glass folder
for f in train_crops/bottle/glass*/*; do
    cp -v "$f" train/glass/"${f//\//_}"
done

# add to cardboard folder

for f in train_crops/cardboard/*/*; do
    cp -v "$f" train/cardboard/"${f//\//_}"
done

# add to metal folder
for f in train_crops/cans/*/*; do
    cp -v "$f" train/metal/"${f//\//_}"
done

mkdir test
cd test

mkdir plastic
mkdir glass 
mkdir cardboard 
mkdir metal
cd .. 

# add to plastic folder
for f in test_crops/bottle/bottle*/*; do
    cp -v "$f" test/plastic/"${f//\//_}"
done

for f in ttest_crops/canister/*/*; do
    cp -v "$f" test/plastic/"${f//\//_}"
done

for f in test_crops/detergent/detergent*/*; do
    cp -v "$f" test/plastic/"${f//\//_}"
done

# add to glass folder
for f in test_crops/bottle/glass*/*; do
    cp -v "$f" test/glass/"${f//\//_}"
done

# add to cardboard folder

for f in test_crops/cardboard/*/*; do
    cp -v "$f" test/cardboard/"${f//\//_}"
done

# add to metal folder

for f in test_crops/cans/*/*; do
    cp -v "$f" test/metal/"${f//\//_}"
done

# remove old folders and files that are not used for this project
rm -r test_crops
rm -r train_crops 
rm Classification.md
rm move.sh