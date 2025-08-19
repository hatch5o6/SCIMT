MYPONY=black
HISPONY=blue

if [ $MYPONY != $HISPONY ]
then
    echo "$MYPONY is not $HISPONY"
fi

if [ $MYPONY = 'black' ]
then
    echo "MYPONY is black"
else
    echo "hey what?"
fi