dirlist=$1

for curdir in $(cat $dirlist)
do
python eval.py --dir /home/nfarrugi/Documents/datasets/BumbleBuzz/BumbleBuzz/ --name $curdir
done