# Declare ad pages
declare -a AdPage=("pr1954_p0939_0.json" )

Path=${Path}

echo "Removing Empty Page"

for json in "${AdPage[@]}"; do
  echo "Removing ${Path}$json"
  rm ${Path}$json
done