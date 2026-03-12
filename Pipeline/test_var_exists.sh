NAME_SET=crackers

if [[ ! -v NAME_SET ]]; then
  echo "NAME_SET does not exist"
else
  echo "NAME_SET is set: $NAME_SET"
fi