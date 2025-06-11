
config=$1
sh Pipeline/train_SC.sh $config
sh Pipeline/pred_SC.sh $config