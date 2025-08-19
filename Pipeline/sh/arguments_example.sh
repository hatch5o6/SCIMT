# NO_GROUPING=false
# COGNATE_THRESH=0.5
# SC_MODEL_TYPE=RNN
# SEED=0
# VAL_COGNATES="null"
# TEST_COGNATES="null"
# while getopts "s:t:p:gm:z:r:v:h:" opt; do
#     case ${opt} in 
#         s)  SRC=$OPTARG;;
#         t)  TGT=$OPTARG;;
#         p)  COGNATE_TRAIN=$OPTARG;;
#         g)  NO_GROUPING=true;;
#         m)  SC_MODEL_TYPE=$OPTARG;;
#         z)  SEED=$OPTARG;;
#         r)  RNN_HYPERPARAMS=$OPTARG;;
#         v)  VAL_COGNATES;;
#         h)  TEST_COGNATES;;
#     esac
# done