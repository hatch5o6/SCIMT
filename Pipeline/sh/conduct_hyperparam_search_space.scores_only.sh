for f in /home/hatch5o6/Cognate/code/Pipeline/cfg/SC-HYPERPARAM_SEARCH/*
do
    echo Calculating for $f
    Pipeline/train_SC.4.3.calculate.sh $f 
done

for f in /home/hatch5o6/Cognate/code/Pipeline/cfg/SC_SMT/*
do
    echo Calculating for $F
    Pipeline/train_SC.4.3.calculate.sh $f
done